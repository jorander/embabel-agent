/*
 * Copyright 2024-2026 Embabel Pty Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.embabel.agent.spi.support

import com.embabel.agent.api.annotation.support.Wumpus
import com.embabel.agent.api.common.InteractionId
import com.embabel.agent.api.tool.ToolObject
import com.embabel.agent.core.AgentProcess
import com.embabel.agent.core.Blackboard
import com.embabel.agent.core.ProcessContext
import com.embabel.agent.core.ProcessOptions
import com.embabel.agent.core.internal.LlmOperations
import com.embabel.agent.core.support.InvalidLlmReturnFormatException
import com.embabel.agent.core.support.InvalidLlmReturnTypeException
import com.embabel.agent.core.support.LlmInteraction
import com.embabel.agent.core.support.safelyGetToolsFrom
import com.embabel.agent.spi.streaming.StreamingLlmOperations
import com.embabel.agent.spi.support.springai.ChatClientLlmOperations
import com.embabel.agent.spi.support.springai.SpringAiLlmService
import com.embabel.agent.spi.validation.DefaultValidationPromptGenerator
import com.embabel.agent.support.SimpleTestAgent
import com.embabel.agent.test.common.EventSavingAgenticEventListener
import com.embabel.chat.SystemMessage
import com.embabel.chat.UserMessage
import com.embabel.common.ai.model.DefaultOptionsConverter
import com.embabel.common.ai.model.LlmOptions
import com.embabel.common.ai.model.ModelProvider
import com.embabel.common.ai.model.ModelSelectionCriteria
import com.embabel.common.textio.template.JinjavaTemplateRenderer
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import io.mockk.every
import io.mockk.mockk
import io.mockk.slot
import jakarta.validation.Validation
import jakarta.validation.constraints.Pattern
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory
import org.springframework.ai.chat.messages.AssistantMessage
import org.springframework.ai.chat.model.ChatModel
import org.springframework.ai.chat.model.ChatResponse
import org.springframework.ai.chat.model.Generation
import org.springframework.ai.chat.prompt.ChatOptions
import org.springframework.ai.chat.prompt.DefaultChatOptions
import org.springframework.ai.chat.prompt.Prompt
import org.springframework.ai.model.tool.ToolCallingChatOptions
import reactor.core.publisher.Flux
import reactor.test.StepVerifier
import java.time.Duration
import java.time.LocalDate
import java.util.concurrent.Executors
import java.util.function.Predicate
import kotlin.test.assertEquals

/**
 * Fake ChatModel with fixed response that captures prompts
 * and tools passed to it.
 * @param responses if > 1 element, they'll be returned in turn.
 * Otherwise, the single response will be returned on every request
 */
class FakeChatModel(
    val responses: List<String>,
    private val options: ChatOptions = DefaultChatOptions(),
) : ChatModel {

    constructor(
        response: String,
        options: ChatOptions = DefaultChatOptions(),
    ) : this(
        listOf(response), options
    )

    val response: String get() = responses.single()

    private var index = 0

    val promptsPassed = mutableListOf<Prompt>()
    val optionsPassed = mutableListOf<ToolCallingChatOptions>()

    override fun getDefaultOptions(): ChatOptions = options

    override fun call(prompt: Prompt): ChatResponse {
        promptsPassed.add(prompt)
        val options = prompt.options as? ToolCallingChatOptions
            ?: throw IllegalArgumentException("Expected ToolCallingChatOptions")
        optionsPassed.add(options)
        return ChatResponse(
            listOf(
                Generation(AssistantMessage(responses[index])).also {
                    // If we have more than one response, step through them
                    if (responses.size > 1) ++index
                }
            )
        )
    }

    override fun stream(prompt: Prompt): Flux<ChatResponse?> {
        promptsPassed.add(prompt)
        val options = prompt.options as? ToolCallingChatOptions
            ?: throw IllegalArgumentException("Expected ToolCallingChatOptions")
        optionsPassed.add(options)
        return Flux.fromIterable(responses)
                .map { response ->
                    ChatResponse(
                        listOf(
                            Generation(AssistantMessage(response))
                        )
                    )
                }
    }
}


class ChatClientLlmOperationsTest {

    private val logger = LoggerFactory.getLogger(ChatClientLlmOperationsTest::class.java)

    data class Setup(
        val llmOperations: LlmOperations,
        val streamingLlmOperations: StreamingLlmOperations,
        val mockAgentProcess: AgentProcess,
        val mutableLlmInvocationHistory: MutableLlmInvocationHistory,
    )

    private fun createChatClientLlmOperations(
        fakeChatModel: FakeChatModel,
        dataBindingProperties: LlmDataBindingProperties = LlmDataBindingProperties(),
        useMessageStreamer: Boolean = false,
    ): Setup {
        val ese = EventSavingAgenticEventListener()
        val mutableLlmInvocationHistory = MutableLlmInvocationHistory()
        val mockProcessContext = mockk<ProcessContext>()
        every { mockProcessContext.platformServices } returns mockk()
        every { mockProcessContext.platformServices.agentPlatform } returns mockk()
        every { mockProcessContext.platformServices.agentPlatform.toolGroupResolver } returns RegistryToolGroupResolver(
            "mt",
            emptyList()
        )
        every { mockProcessContext.platformServices.eventListener } returns ese
        every { mockProcessContext.processOptions } returns ProcessOptions()
        val mockAgentProcess = mockk<AgentProcess>()
        every { mockAgentProcess.recordLlmInvocation(any()) } answers {
            mutableLlmInvocationHistory.invocations.add(firstArg())
        }
        every { mockProcessContext.onProcessEvent(any()) } answers { ese.onProcessEvent(firstArg()) }
        every { mockProcessContext.agentProcess } returns mockAgentProcess

        every { mockAgentProcess.agent } returns SimpleTestAgent
        every { mockAgentProcess.processContext } returns mockProcessContext

        // Add blackboard for guardrail validation (defensive - returns null if not needed)
        val blackboard = mockk<com.embabel.agent.core.Blackboard>(relaxed = true)
        every { mockAgentProcess.blackboard } returns blackboard

        val mockModelProvider = mockk<ModelProvider>()
        val crit = slot<ModelSelectionCriteria>()
        val fakeLlm = SpringAiLlmService("fake", "provider", fakeChatModel, DefaultOptionsConverter)
        every { mockModelProvider.getLlm(capture(crit)) } returns fakeLlm
        val cco = ChatClientLlmOperations(
            modelProvider = mockModelProvider,
            toolDecorator = DefaultToolDecorator(),
            validator = Validation.buildDefaultValidatorFactory().validator,
            validationPromptGenerator = DefaultValidationPromptGenerator(),
            templateRenderer = JinjavaTemplateRenderer(),
            objectMapper = jacksonObjectMapper().registerModule(JavaTimeModule()),
            dataBindingProperties = dataBindingProperties,
            asyncer = ExecutorAsyncer(Executors.newCachedThreadPool()),
            useMessageStreamer = useMessageStreamer
        )
        return Setup(cco, cco, mockAgentProcess, mutableLlmInvocationHistory)
    }

    data class Dog(val name: String)

    data class TemporalDog(
        val name: String,
        val birthDate: LocalDate,
    )

    @Nested
    inner class CreateObject {

        @Test
        fun `passes correct prompt`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(fakeChatModel)
            setup.llmOperations.createObject(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            val promptText = fakeChatModel.promptsPassed[0].toString()
            assertTrue(promptText.contains("\$schema"), "Prompt contains JSON schema")
            assertTrue(promptText.contains(promptText), "Prompt contains user prompt:\n$promptText")
        }

        @Test
        fun `returns string`() {
            val fakeChatModel = FakeChatModel("fake response")

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObject(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = String::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(fakeChatModel.response, result)
        }

        @Test
        fun `handles ill formed JSON when returning data class`() {
            val fakeChatModel = FakeChatModel("This ain't no JSON")

            val setup = createChatClientLlmOperations(fakeChatModel)
            try {
                setup.llmOperations.createObject(
                    messages = listOf(UserMessage("prompt")),
                    interaction = LlmInteraction(
                        id = InteractionId("id"), llm = LlmOptions()
                    ),
                    outputClass = Dog::class.java,
                    action = SimpleTestAgent.actions.first(),
                    agentProcess = setup.mockAgentProcess,
                )
                fail("Should have thrown exception")
            } catch (e: InvalidLlmReturnFormatException) {
                assertEquals(fakeChatModel.response, e.llmReturn)
                assertTrue(e.infoString(verbose = true).contains(fakeChatModel.response))
            }
        }

        @Test
        fun `returns data class`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObject(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(duke, result)
        }

        @Test
        fun `passes JSON few shot example`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObject(
                messages = listOf(
                    UserMessage(
                        """
                    Return a dog. Dogs look like this:
                {
                    "name": "Duke",
                    "type": "Dog"
                }
                """.trimIndent()
                    )
                ),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(duke, result)
        }

        @Test
        fun `presents no tools to ChatModel`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObject(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(duke, result)
            assertEquals(1, fakeChatModel.promptsPassed.size)
            val tools = fakeChatModel.optionsPassed[0].toolCallbacks
            assertEquals(0, tools.size)
        }

        @Test
        fun `presents tools to ChatModel via doTransform`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            // Wumpus's have tools
            val tools = safelyGetToolsFrom(ToolObject(Wumpus("wumpy")))
            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.doTransform(
                messages = listOf(
                    SystemMessage("do whatever"),
                    UserMessage("prompt"),
                ),
                interaction = LlmInteraction(
                    id = InteractionId("id"),
                    llm = LlmOptions(),
                    tools = tools,
                ),
                outputClass = Dog::class.java,
                llmRequestEvent = null,
            )
            assertEquals(duke, result)
            assertEquals(1, fakeChatModel.promptsPassed.size)
            val passedTools = fakeChatModel.optionsPassed[0].toolCallbacks
            assertEquals(tools.size, passedTools.size, "Must have passed same number of tools")
            assertEquals(
                tools.map { it.definition.name }.toSet(),
                passedTools.map { it.toolDefinition.name() }.toSet(),
            )
        }

        @Test
        fun `presents tools to ChatModel when given multiple messages`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            // Wumpus's have tools - use native Tool interface
            val tools = safelyGetToolsFrom(ToolObject(Wumpus("wumpy")))
            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObject(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"),
                    llm = LlmOptions(),
                    tools = tools,
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(duke, result)
            assertEquals(1, fakeChatModel.promptsPassed.size)
            val passedTools = fakeChatModel.optionsPassed[0].toolCallbacks
            assertEquals(tools.size, passedTools.size, "Must have passed same number of tools")
            assertEquals(
                tools.map { it.definition.name }.sorted(),
                passedTools.map { it.toolDefinition.name() })
        }

        @Test
        fun `handles reasoning model return`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(
                "<think>Deep thoughts</think>\n" + jacksonObjectMapper().writeValueAsString(duke)
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObject(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(duke, result)
        }

        @Test
        fun `handles LocalDate return`() {
            val duke = TemporalDog("Duke", birthDate = LocalDate.of(2021, 2, 26))

            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().registerModule(JavaTimeModule()).writeValueAsString(duke)
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObject(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = TemporalDog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(duke, result)
        }
    }

    @Nested
    inner class CreateObjectStream {

        @Test
        fun `passes correct prompt`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(fakeChatModel)
            setup.streamingLlmOperations.createObjectStream(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()

            val promptText = fakeChatModel.promptsPassed[0].toString()
            assertTrue(promptText.contains("\$schema"), "Prompt contains JSON schema")
            assertTrue(promptText.contains(promptText), "Prompt contains user prompt:\n$promptText")
        }

        @Test
        fun `handles ill formed JSON when returning data class`() {
            val fakeChatModel = FakeChatModel("This ain't no JSON")

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStream(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            StepVerifier.create(result)
                .verifyComplete()  // No data returned
        }

        @Test
        fun `returns data class`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStream(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(duke, result)
        }

        @Test
        fun `passes JSON few shot example`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStream(
                messages = listOf(
                    UserMessage(
                        """
                    Return a dog. Dogs look like this:
                {
                    "name": "Duke",
                    "type": "Dog"
                }
                """.trimIndent()
                    )
                ),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(duke, result)
        }

        @Test
        fun `presents no tools to ChatModel`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStream(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(duke, result)
            assertEquals(1, fakeChatModel.promptsPassed.size)
            val tools = fakeChatModel.optionsPassed[0].toolCallbacks
            assertEquals(0, tools.size)
        }

        @Test
        fun `presents tools to ChatModel via doTransform`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            // Wumpus's have tools
            val tools = safelyGetToolsFrom(ToolObject(Wumpus("wumpy")))
            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.doTransformObjectStream(
                messages = listOf(
                    SystemMessage("do whatever"),
                    UserMessage("prompt"),
                ),
                interaction = LlmInteraction(
                    id = InteractionId("id"),
                    llm = LlmOptions(),
                    tools = tools,
                ),
                outputClass = Dog::class.java,
                llmRequestEvent = null,
            ).blockLast()
            assertEquals(duke, result)
            assertEquals(1, fakeChatModel.promptsPassed.size)
            val passedTools = fakeChatModel.optionsPassed[0].toolCallbacks
            assertEquals(tools.size, passedTools.size, "Must have passed same number of tools")
            assertEquals(
                tools.map { it.definition.name }.toSet(),
                passedTools.map { it.toolDefinition.name() }.toSet(),
            )
        }

        @Test
        fun `presents tools to ChatModel when given multiple messages`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            // Wumpus's have tools - use native Tool interface
            val tools = safelyGetToolsFrom(ToolObject(Wumpus("wumpy")))
            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStream(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"),
                    llm = LlmOptions(),
                    tools = tools,
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(duke, result)
            assertEquals(1, fakeChatModel.promptsPassed.size)
            val passedTools = fakeChatModel.optionsPassed[0].toolCallbacks
            assertEquals(tools.size, passedTools.size, "Must have passed same number of tools")
            assertEquals(
                tools.map { it.definition.name }.sorted(),
                passedTools.map { it.toolDefinition.name() })
        }

        @Test
        fun `handles reasoning model return`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(
                "<think>Deep thoughts</think>\n" + jacksonObjectMapper().writeValueAsString(duke)
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStream(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(duke, result)
        }

        @Test
        fun `handles LocalDate return`() {
            val duke = TemporalDog("Duke", birthDate = LocalDate.of(2021, 2, 26))

            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().registerModule(JavaTimeModule()).writeValueAsString(duke)
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStream(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = TemporalDog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(duke, result)
        }
    }

    data class TestItem(val name: String, val value: Int)

    @Nested
    inner class CreateObjectStreamWithThinking {

        @Test
        fun `should handle single complete chunk`() {
            val fakeChatModel = FakeChatModel("<think>This is thinking content</think>\n")
            val setup = createChatClientLlmOperations(fakeChatModel)

            // When
            val result = setup.streamingLlmOperations.createObjectStreamWithThinking(
                messages = listOf(UserMessage("test")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = TestItem::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess
            )


            // Then: Should emit one thinking event for the complete line
            StepVerifier.create(result)
                .expectNextMatches {
                    it.isThinking() && it.getThinking() == "This is thinking content"
                }
                .expectComplete()
                .verify(Duration.ofSeconds(1))
        }

        @Test
        fun `should handle multi-chunk JSONL object stream`() {
            // Given: Multiple chunks forming JSONL objects
            val chunks = listOf(
                "{\"name\":\"Item1\",\"value\":",
                "42}\n{\"name\":\"Item2\",",
                "\"value\":84}\n"
            )
            val fakeChatModel = FakeChatModel(chunks)
            val setup = createChatClientLlmOperations(fakeChatModel)

            // When
            val result = setup.streamingLlmOperations.createObjectStreamWithThinking(
                messages = listOf(UserMessage("test")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = TestItem::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess
            )

            // Then: Should emit two object events
            StepVerifier.create(result)
                .expectNextMatches {
                    it.isObject() && it.getObject()?.name == "Item1" && it.getObject()?.value == 42
                }
                .expectNextMatches {
                    it.isObject() && it.getObject()?.name == "Item2" && it.getObject()?.value == 84
                }
                .expectComplete()
                .verify(Duration.ofSeconds(1))
        }

        @Test
        fun `should handle mixed thinking and object content in chunks`() {
            // Given: Realistic chunking that splits thinking and JSON across chunk boundaries
            val chunks = listOf(
                "<think>Ana",                  // Partial thinking start
                "lyzing req",                          // Partial thinking middle
                "uirement</think>\n{\"name\":",        // Thinking end + partial JSON
                "\"TestItem\",\"va",                   // Partial JSON middle
                "lue\":123}\n<think>Done",             // JSON end + partial thinking
                "</think>\n"                           // Thinking end
            )
            val fakeChatModel = FakeChatModel(chunks)
            val setup = createChatClientLlmOperations(fakeChatModel)

            // When
            val result = setup.streamingLlmOperations.createObjectStreamWithThinking(
                messages = listOf(UserMessage("test")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = TestItem::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess
            )

            // Then: Should emit thinking, object, thinking in correct order
            StepVerifier.create(result)
                .expectNextMatches {
                    it.isThinking() && it.getThinking() == "Analyzing requirement"
                }
                .expectNextMatches {
                    it.isObject() && it.getObject()?.name == "TestItem" && it.getObject()?.value == 123
                }
                .expectNextMatches {
                    it.isThinking() && it.getThinking() == "Done"
                }
                .expectComplete()
                .verify(Duration.ofSeconds(1))
        }

        @Test
        fun `should handle real streaming with reactive callbacks`() {
            // Given: Mixed content with multiple events
            val chunks = listOf(
                "<think>Processing request</think>\n",
                "{\"name\":\"Item1\",\"value\":100}\n",
                "{\"name\":\"Item2\",\"value\":200}\n",
                "<think>Request completed</think>\n"
            )
            val fakeChatModel = FakeChatModel(chunks)
            val setup = createChatClientLlmOperations(fakeChatModel)

            // When: Subscribe with real reactive callbacks
            val receivedEvents = mutableListOf<String>()
            var errorOccurred: Throwable? = null
            var completionCalled = false

            val result = setup.streamingLlmOperations.createObjectStreamWithThinking(
                messages = listOf(UserMessage("test")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = TestItem::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess
            )

            result
                .doOnNext { event ->
                    when {
                        event.isThinking() -> {
                            val content = event.getThinking()!!
                            receivedEvents.add("THINKING: $content")
                            logger.info("Received thinking: {}", content)
                        }

                        event.isObject() -> {
                            val obj = event.getObject()!!
                            receivedEvents.add("OBJECT: ${obj.name}=${obj.value}")
                            logger.info("Received object: {}={}", obj.name, obj.value)
                        }
                    }
                }
                .doOnError { error ->
                    errorOccurred = error
                    logger.error("Stream error: {}", error.message)
                }
                .doOnComplete {
                    completionCalled = true
                    logger.info("Stream completed successfully")
                }
                .subscribe()

            // Give stream time to complete
            Thread.sleep(500)

            // Then: Verify real reactive behavior
            assertNull(errorOccurred, "No errors should occur")
            assertTrue(completionCalled, "Stream should complete successfully")
            assertEquals(4, receivedEvents.size, "Should receive all events")
            assertEquals("THINKING: Processing request", receivedEvents[0])
            assertEquals("OBJECT: Item1=100", receivedEvents[1])
            assertEquals("OBJECT: Item2=200", receivedEvents[2])
            assertEquals("THINKING: Request completed", receivedEvents[3])
        }
    }

    @Nested
    inner class CreateObjectIfPossible {

        @Test
        fun `should have correct prompt with success and failure`() {
            val fakeChatModel =
                FakeChatModel(
                    jacksonObjectMapper().writeValueAsString(
                        MaybeReturn<Dog>(
                            failure = "didn't work"
                        )
                    )
                )

            val prompt = "The quick brown fox jumped over the lazy dog"
            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObjectIfPossible(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertTrue(result.isFailure)
            val promptText = fakeChatModel.promptsPassed[0].toString()
            assertTrue(promptText.contains("\$schema"), "Prompt contains JSON schema")
            assertTrue(promptText.contains(promptText), "Prompt contains user prompt:\n$promptText")

            assertTrue(promptText.contains("possible"), "Prompt mentions possible")
            assertTrue(promptText.contains("success"), "Prompt mentions success")
            assertTrue(promptText.contains("failure"), "Prompt mentions failure")
        }

        @Test
        fun `returns data class - success`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().writeValueAsString(
                    MaybeReturn(
                        success = duke
                    )
                )
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObjectIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(duke, result.getOrThrow())
        }

        @Test
        fun `handles reasoning model success return`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(
                "<think>More deep thoughts</think>\n" + jacksonObjectMapper().writeValueAsString(
                    MaybeReturn(
                        success = duke
                    )
                )
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObjectIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(duke, result.getOrThrow())
        }

        @Test
        fun `handles LocalDate return`() {
            val duke = TemporalDog("Duke", birthDate = LocalDate.of(2021, 2, 26))

            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().registerModule(JavaTimeModule()).writeValueAsString(
                    MaybeReturn(duke)
                )
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObjectIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = TemporalDog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(duke, result.getOrThrow())
        }

        @Test
        fun `handles ill formed JSON when returning data class`() {
            val fakeChatModel = FakeChatModel("This ain't no JSON")

            val setup = createChatClientLlmOperations(fakeChatModel)
            try {
                setup.llmOperations.createObjectIfPossible(
                    messages = listOf(UserMessage("prompt")),
                    interaction = LlmInteraction(
                        id = InteractionId("id"), llm = LlmOptions()
                    ),
                    outputClass = Dog::class.java,
                    action = SimpleTestAgent.actions.first(),
                    agentProcess = setup.mockAgentProcess,
                )
                fail("Should have thrown exception")
            } catch (e: InvalidLlmReturnFormatException) {
                assertEquals(fakeChatModel.response, e.llmReturn)
                assertTrue(e.infoString(verbose = true).contains(fakeChatModel.response))
            }
        }

        @Test
        fun `returns data class - failure`() {
            val fakeChatModel =
                FakeChatModel(
                    jacksonObjectMapper().writeValueAsString(
                        MaybeReturn<Dog>(
                            failure = "didn't work"
                        )
                    )
                )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.llmOperations.createObjectIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertTrue(result.isFailure)
        }

        @Test
        fun `presents tools to ChatModel`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().writeValueAsString(
                    MaybeReturn(duke)
                )
            )

            // Wumpus's have tools - use native Tool interface
            val tools = safelyGetToolsFrom(ToolObject(Wumpus("wumpy")))
            val setup = createChatClientLlmOperations(fakeChatModel)
            setup.llmOperations.createObjectIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"),
                    llm = LlmOptions(),
                    tools = tools,
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(1, fakeChatModel.promptsPassed.size)
            val passedTools = fakeChatModel.optionsPassed[0].toolCallbacks
            assertEquals(tools.size, passedTools.size, "Must have passed same number of tools")
            assertEquals(
                tools.map { it.definition.name }.sorted(),
                passedTools.map { it.toolDefinition.name() })
        }
    }

    @Nested
    inner class CreateObjectStreamIfPossible {

        @Test
        @Disabled("createObjectStreamIfPossible does not have an implemenation with a specific prompt yet")
        fun `should have correct prompt with success and failure`() {
            val fakeChatModel =
                FakeChatModel(
                    jacksonObjectMapper().writeValueAsString(
                        MaybeReturn<Dog>(
                            failure = "didn't work"
                        )
                    )
                )

            val prompt = "The quick brown fox jumped over the lazy dog"
            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStreamIfPossible(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertTrue(result!!.isFailure)
            val promptText = fakeChatModel.promptsPassed[0].toString()
            assertTrue(promptText.contains("\$schema"), "Prompt contains JSON schema")
            assertTrue(promptText.contains(promptText), "Prompt contains user prompt:\n$promptText")

            assertTrue(promptText.contains("possible"), "Prompt mentions possible")
            assertTrue(promptText.contains("success"), "Prompt mentions success")
            assertTrue(promptText.contains("failure"), "Prompt mentions failure")
        }

        @Test
        fun `returns data class - success`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().writeValueAsString(duke)
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStreamIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(duke, result!!.getOrThrow())
        }

        @Test
        fun `handles reasoning model success return`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(
                "<think>More deep thoughts</think>\n" + jacksonObjectMapper().writeValueAsString(duke)
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStreamIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(duke, result!!.getOrThrow())
        }

        @Test
        fun `handles LocalDate return`() {
            val duke = TemporalDog("Duke", birthDate = LocalDate.of(2021, 2, 26))

            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().registerModule(JavaTimeModule()).writeValueAsString(duke)
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStreamIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = TemporalDog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(duke, result!!.getOrThrow())
        }

        @Test
        fun `handles ill formed JSON when returning data class`() {
            val fakeChatModel = FakeChatModel("This ain't no JSON")

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStreamIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            StepVerifier.create(result)
                .verifyComplete() // No data returned
        }

        @Test
        fun `returns data class - failure`() {
            val fakeChatModel =
                FakeChatModel(
                    jacksonObjectMapper().writeValueAsString(
                        MaybeReturn<Dog>(
                            failure = "didn't work"
                        )
                    )
                )

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.createObjectStreamIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertTrue(result!!.isFailure)
        }

        @Test
        fun `presents tools to ChatModel`() {
            val duke = Dog("Duke")

            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().writeValueAsString(
                    MaybeReturn(duke)
                )
            )

            // Wumpus's have tools - use native Tool interface
            val tools = safelyGetToolsFrom(ToolObject(Wumpus("wumpy")))
            val setup = createChatClientLlmOperations(fakeChatModel)
            setup.streamingLlmOperations.createObjectStreamIfPossible(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"),
                    llm = LlmOptions(),
                    tools = tools,
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()
            assertEquals(1, fakeChatModel.promptsPassed.size)
            val passedTools = fakeChatModel.optionsPassed[0].toolCallbacks
            assertEquals(tools.size, passedTools.size, "Must have passed same number of tools")
            assertEquals(
                tools.map { it.definition.name }.sorted(),
                passedTools.map { it.toolDefinition.name() })
        }
    }

    @Nested
    inner class GenerateStream {

        @Test
        fun `returns string`() {
            val fakeChatModel = FakeChatModel("fake response")

            val setup = createChatClientLlmOperations(fakeChatModel)
            val result = setup.streamingLlmOperations.generateStream(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            ).blockLast()

            assertEquals(fakeChatModel.response, result)
        }
    }

    @Nested
    inner class TimeoutBehavior {

        /**
         * Fake ChatModel that introduces a delay before returning.
         * Used to test timeout behavior.
         */
        inner class DelayingFakeChatModel(
            private val response: String,
            private val delayMillis: Long,
            options: ChatOptions = DefaultChatOptions(),
        ) : ChatModel {
            private val defaultOptions = options
            val callCount = java.util.concurrent.atomic.AtomicInteger(0)

            override fun getDefaultOptions(): ChatOptions = defaultOptions

            override fun call(prompt: Prompt): ChatResponse {
                callCount.incrementAndGet()
                Thread.sleep(delayMillis)
                val options = prompt.options as? ToolCallingChatOptions
                    ?: throw IllegalArgumentException("Expected ToolCallingChatOptions")
                return ChatResponse(listOf(Generation(AssistantMessage(response))))
            }
        }

        @Test
        fun `should timeout when LLM call exceeds timeout`() {
            val duke = Dog("Duke")
            val delayingChatModel = DelayingFakeChatModel(
                response = jacksonObjectMapper().writeValueAsString(duke),
                delayMillis = 500,
            )

            val setup = createChatClientLlmOperationsWithDelayingModel(delayingChatModel)

            val exception = assertThrows(RuntimeException::class.java) {
                setup.llmOperations.createObject(
                    messages = listOf(UserMessage("Give me a dog")),
                    interaction = LlmInteraction(
                        id = InteractionId("timeout-test"),
                        llm = LlmOptions().withTimeout(java.time.Duration.ofMillis(100)),
                    ),
                    outputClass = Dog::class.java,
                    action = SimpleTestAgent.actions.first(),
                    agentProcess = setup.mockAgentProcess,
                )
            }

            assertTrue(
                exception.message?.contains("timed out") == true ||
                        exception.cause is java.util.concurrent.TimeoutException,
                "Should have timed out, but got: ${exception.message}"
            )
        }

        private fun createChatClientLlmOperationsWithDelayingModel(
            delayingChatModel: DelayingFakeChatModel,
        ): Setup {
            val ese = EventSavingAgenticEventListener()
            val mutableLlmInvocationHistory = MutableLlmInvocationHistory()
            val mockProcessContext = mockk<ProcessContext>()
            every { mockProcessContext.platformServices } returns mockk()
            every { mockProcessContext.platformServices.agentPlatform } returns mockk()
            every { mockProcessContext.platformServices.agentPlatform.toolGroupResolver } returns RegistryToolGroupResolver(
                "mt",
                emptyList()
            )
            every { mockProcessContext.platformServices.eventListener } returns ese
            every { mockProcessContext.processOptions } returns ProcessOptions()
            val mockAgentProcess = mockk<AgentProcess>()
            every { mockAgentProcess.recordLlmInvocation(any()) } answers {
                mutableLlmInvocationHistory.invocations.add(firstArg())
            }
            every { mockProcessContext.onProcessEvent(any()) } answers { ese.onProcessEvent(firstArg()) }
            every { mockProcessContext.agentProcess } returns mockAgentProcess

            every { mockAgentProcess.agent } returns SimpleTestAgent
            every { mockAgentProcess.processContext } returns mockProcessContext

            // Add blackboard for guardrail validation
            val blackboard = mockk<Blackboard>(relaxed = true)
            every { mockAgentProcess.blackboard } returns blackboard

            val mockModelProvider = mockk<ModelProvider>()
            val crit = slot<ModelSelectionCriteria>()
            val fakeLlm = SpringAiLlmService("fake", "provider", delayingChatModel, DefaultOptionsConverter)
            every { mockModelProvider.getLlm(capture(crit)) } returns fakeLlm
            val promptsProperties = LlmOperationsPromptsProperties().apply {
                defaultTimeout = java.time.Duration.ofMillis(100)  // Short default timeout
            }
            val cco = ChatClientLlmOperations(
                modelProvider = mockModelProvider,
                toolDecorator = DefaultToolDecorator(),
                validator = Validation.buildDefaultValidatorFactory().validator,
                validationPromptGenerator = DefaultValidationPromptGenerator(),
                templateRenderer = JinjavaTemplateRenderer(),
                objectMapper = jacksonObjectMapper().registerModule(JavaTimeModule()),
                dataBindingProperties = LlmDataBindingProperties(maxAttempts = 1),  // No retries for timeout tests
                llmOperationsPromptsProperties = promptsProperties,
                asyncer = ExecutorAsyncer(Executors.newCachedThreadPool()),
            )
            return Setup(cco, cco, mockAgentProcess, mutableLlmInvocationHistory)
        }
    }

    @Nested
    inner class RetryOnInvalidJson {

        @Test
        fun `should retry on invalid JSON and succeed`() {
            val duke = Dog("Duke")

            // First response is invalid JSON, second is valid
            val fakeChatModel = FakeChatModel(
                responses = listOf(
                    "This ain't no JSON - malformed response",
                    jacksonObjectMapper().writeValueAsString(duke)
                )
            )

            val setup = createChatClientLlmOperations(
                fakeChatModel,
                LlmDataBindingProperties(maxAttempts = 3)
            )

            val result = setup.llmOperations.createObject(
                messages = listOf(UserMessage("Give me a dog")),
                interaction = LlmInteraction(
                    id = InteractionId("retry-test"),
                    llm = LlmOptions(),
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            assertEquals(duke, result, "Should have retried and got valid response")
            assertEquals(2, fakeChatModel.promptsPassed.size, "Should have made 2 attempts")
        }
    }

    /**
     * Tests for proper system message ordering.
     * Validates fix for GitHub issue #1295: System messages should be consolidated
     * at the beginning of the conversation, not scattered throughout.
     * This is required for:
     * - OpenAI best practices (prevents instruction drift)
     * - DeepSeek compatibility (strict message ordering requirements)
     * - General cross-model reliability
     */
    @Nested
    inner class SystemMessageOrdering {

        @Test
        fun `system message appears only at the beginning of prompt`() {
            val duke = Dog("Duke")
            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val setup = createChatClientLlmOperations(fakeChatModel)
            setup.llmOperations.createObject(
                messages = listOf(UserMessage("Give me a dog named Duke")),
                interaction = LlmInteraction(
                    id = InteractionId("system-ordering-test"),
                    llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            assertEquals(1, fakeChatModel.promptsPassed.size)
            val prompt = fakeChatModel.promptsPassed[0]
            val messages = prompt.instructions

            // Count system messages
            val systemMessages = messages.filterIsInstance<org.springframework.ai.chat.messages.SystemMessage>()
            assertTrue(
                systemMessages.size <= 1,
                "Should have at most one system message, but found ${systemMessages.size}"
            )

            // If there's a system message, it should be first
            if (systemMessages.isNotEmpty()) {
                assertTrue(
                    messages.first() is org.springframework.ai.chat.messages.SystemMessage,
                    "System message should be at the beginning of the prompt"
                )
            }
        }

        @Test
        fun `schema format is included in system message not appended after`() {
            val duke = Dog("Duke")
            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val setup = createChatClientLlmOperations(fakeChatModel)
            setup.llmOperations.createObject(
                messages = listOf(UserMessage("Give me a dog")),
                interaction = LlmInteraction(
                    id = InteractionId("schema-in-system-test"),
                    llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            val prompt = fakeChatModel.promptsPassed[0]
            val messages = prompt.instructions

            // The schema format (containing $schema) should be in the first system message
            val systemMessages = messages.filterIsInstance<org.springframework.ai.chat.messages.SystemMessage>()
            assertTrue(systemMessages.isNotEmpty(), "Should have a system message")

            val firstSystemMessage = systemMessages.first()
            assertTrue(
                firstSystemMessage.text.contains("\$schema") || firstSystemMessage.text.contains("\"type\""),
                "Schema format should be in the system message"
            )

            // Verify no system message appears after user messages
            val userMessageIndex = messages.indexOfFirst { it is org.springframework.ai.chat.messages.UserMessage }
            if (userMessageIndex >= 0) {
                val messagesAfterUser = messages.drop(userMessageIndex + 1)
                val systemMessagesAfterUser =
                    messagesAfterUser.filterIsInstance<org.springframework.ai.chat.messages.SystemMessage>()
                assertTrue(
                    systemMessagesAfterUser.isEmpty(),
                    "No system messages should appear after user messages, but found ${systemMessagesAfterUser.size}"
                )
            }
        }

        @Test
        fun `createObjectIfPossible consolidates system messages`() {
            val duke = Dog("Duke")
            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().writeValueAsString(
                    MaybeReturn(success = duke)
                )
            )

            val setup = createChatClientLlmOperations(fakeChatModel)
            setup.llmOperations.createObjectIfPossible(
                messages = listOf(UserMessage("Give me a dog if possible")),
                interaction = LlmInteraction(
                    id = InteractionId("maybe-return-system-test"),
                    llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            val prompt = fakeChatModel.promptsPassed[0]
            val messages = prompt.instructions

            // Count system messages - should be exactly one at the start
            val systemMessages = messages.filterIsInstance<org.springframework.ai.chat.messages.SystemMessage>()
            assertTrue(
                systemMessages.size <= 1,
                "createObjectIfPossible should consolidate to at most one system message, found ${systemMessages.size}"
            )

            // System message should be first
            if (systemMessages.isNotEmpty()) {
                assertTrue(
                    messages.first() is org.springframework.ai.chat.messages.SystemMessage,
                    "System message should be at the beginning"
                )
            }

            // No system messages after user messages
            val firstNonSystemIndex =
                messages.indexOfFirst { it !is org.springframework.ai.chat.messages.SystemMessage }
            if (firstNonSystemIndex >= 0) {
                val messagesAfterFirst = messages.drop(firstNonSystemIndex)
                val lateSystemMessages =
                    messagesAfterFirst.filterIsInstance<org.springframework.ai.chat.messages.SystemMessage>()
                assertTrue(
                    lateSystemMessages.isEmpty(),
                    "No system messages should appear after non-system messages"
                )
            }
        }

        @Test
        fun `prompt contributions and schema are merged into single system message`() {
            val duke = Dog("Duke")
            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))

            val setup = createChatClientLlmOperations(fakeChatModel)
            setup.llmOperations.createObject(
                messages = listOf(
                    SystemMessage("You are a helpful assistant that creates dogs."),
                    UserMessage("Give me a dog named Duke"),
                ),
                interaction = LlmInteraction(
                    id = InteractionId("merged-system-test"),
                    llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            val prompt = fakeChatModel.promptsPassed[0]
            val messages = prompt.instructions

            // Should have exactly one system message
            val systemMessages = messages.filterIsInstance<org.springframework.ai.chat.messages.SystemMessage>()

            // The single system message should contain the schema
            if (systemMessages.isNotEmpty()) {
                val systemContent = systemMessages.first().text
                assertTrue(
                    systemContent.contains("\$schema") || systemContent.contains("\"type\""),
                    "System message should contain schema format"
                )
            }

            // Verify proper ordering: system first, then user/assistant
            var foundNonSystem = false
            for (message in messages) {
                if (message !is org.springframework.ai.chat.messages.SystemMessage) {
                    foundNonSystem = true
                } else if (foundNonSystem) {
                    fail<Unit>("System message found after non-system message - violates message ordering")
                }
            }
        }
    }

    @Nested
    inner class ApiErrorHandling {

        /**
         * ChatModel that throws RuntimeException simulating an API key
         * that lacks access to the configured model (401/403 from OpenAI).
         */
        inner class ErrorThrowingChatModel(
            private val exception: RuntimeException = RuntimeException("401 Unauthorized: Invalid API key")
        ) : ChatModel {
            override fun getDefaultOptions(): ChatOptions = DefaultChatOptions()
            override fun call(prompt: Prompt): ChatResponse = throw exception
        }

        @Test
        fun `throws RuntimeException with message when API key is invalid`() {
            val errorModel = ErrorThrowingChatModel()

            val setup = createChatClientLlmOperations(
                FakeChatModel("unused").also {
                    // We need to set up the infrastructure but use our own model
                },
                LlmDataBindingProperties(maxAttempts = 1),
            )

            // Replace the model provider to use our error-throwing model
            val ese = EventSavingAgenticEventListener()
            val mutableLlmInvocationHistory = MutableLlmInvocationHistory()
            val mockProcessContext = mockk<ProcessContext>()
            every { mockProcessContext.platformServices } returns mockk()
            every { mockProcessContext.platformServices.agentPlatform } returns mockk()
            every { mockProcessContext.platformServices.agentPlatform.toolGroupResolver } returns RegistryToolGroupResolver(
                "mt",
                emptyList()
            )
            every { mockProcessContext.platformServices.eventListener } returns ese
            every { mockProcessContext.processOptions } returns ProcessOptions()
            val mockAgentProcess = mockk<AgentProcess>()
            every { mockAgentProcess.recordLlmInvocation(any()) } answers {
                mutableLlmInvocationHistory.invocations.add(firstArg())
            }
            every { mockProcessContext.onProcessEvent(any()) } answers { ese.onProcessEvent(firstArg()) }
            every { mockProcessContext.agentProcess } returns mockAgentProcess
            every { mockAgentProcess.agent } returns SimpleTestAgent
            every { mockAgentProcess.processContext } returns mockProcessContext
            val blackboard = mockk<Blackboard>(relaxed = true)
            every { mockAgentProcess.blackboard } returns blackboard

            val mockModelProvider = mockk<ModelProvider>()
            val crit = slot<ModelSelectionCriteria>()
            val fakeLlm = SpringAiLlmService("fake", "provider", errorModel, DefaultOptionsConverter)
            every { mockModelProvider.getLlm(capture(crit)) } returns fakeLlm
            val cco = ChatClientLlmOperations(
                modelProvider = mockModelProvider,
                toolDecorator = DefaultToolDecorator(),
                validator = Validation.buildDefaultValidatorFactory().validator,
                validationPromptGenerator = DefaultValidationPromptGenerator(),
                templateRenderer = JinjavaTemplateRenderer(),
                objectMapper = jacksonObjectMapper().registerModule(JavaTimeModule()),
                dataBindingProperties = LlmDataBindingProperties(maxAttempts = 1),
                asyncer = ExecutorAsyncer(Executors.newCachedThreadPool()),
            )

            val exception = assertThrows(RuntimeException::class.java) {
                cco.doTransform(
                    messages = listOf(UserMessage("prompt")),
                    interaction = LlmInteraction(
                        id = InteractionId("api-error-test"),
                        llm = LlmOptions(),
                    ),
                    outputClass = String::class.java,
                    llmRequestEvent = null,
                )
            }
            // Should get a RuntimeException, not an NPE
            assertFalse(
                exception is NullPointerException,
                "Should not be NullPointerException, but got: ${exception::class.simpleName}"
            )
        }
    }

    @Nested
    inner class ReturnValidation {

        @Test
        fun `validates with no rules`() {
            val duke = Dog("Duke")
            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(duke))
            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(fakeChatModel)
            val createdDog = setup.llmOperations.createObject(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = Dog::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            assertEquals(duke, createdDog)
        }

        @Test
        fun `validated field with no violation`() {
            // Picky eater
            data class BorderCollie(
                val name: String,
                @field:Pattern(regexp = "^mince$", message = "eats field must be 'mince'")
                val eats: String,
            )

            // This is OK
            val husky = BorderCollie("Husky", eats = "mince")
            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(husky))
            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(fakeChatModel)
            val createdDog = setup.llmOperations.createObject(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = BorderCollie::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(husky, createdDog)
        }

        @Test
        fun `corrects validated field with violation`() {
            // Picky eater
            data class BorderCollie(
                val name: String,
                @field:Pattern(regexp = "^mince$", message = "eats field must be 'mince'")
                val eats: String,
            )

            val invalidHusky = BorderCollie("Husky", eats = "kibble")
            val validHusky = BorderCollie("Husky", eats = "mince")
            val fakeChatModel = FakeChatModel(
                responses = listOf(
                    jacksonObjectMapper().writeValueAsString(invalidHusky),
                    jacksonObjectMapper().writeValueAsString(validHusky),
                )
            )
            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(fakeChatModel)
            val createdDog = setup.llmOperations.createObject(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = BorderCollie::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            assertEquals(validHusky, createdDog, "Invalid response should have been corrected")
        }

        @Test
        fun `fails to correct validated field with violation`() {
            // Picky eater
            data class BorderCollie(
                val name: String,
                @field:Pattern(regexp = "^mince$", message = "eats field must be 'mince'")
                val eats: String,
            )

            val invalidHusky = BorderCollie("Husky", eats = "kibble")
            val fakeChatModel = FakeChatModel(
                response = jacksonObjectMapper().writeValueAsString(invalidHusky)
            )

            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(fakeChatModel)
            try {
                setup.llmOperations.createObject(
                    messages = listOf(UserMessage(prompt)),
                    interaction = LlmInteraction(
                        id = InteractionId("id"), llm = LlmOptions()
                    ),
                    outputClass = BorderCollie::class.java,
                    action = SimpleTestAgent.actions.first(),
                    agentProcess = setup.mockAgentProcess,
                )
                fail("Should have thrown an exception on invalid object")
            } catch (e: InvalidLlmReturnTypeException) {
                assertEquals(invalidHusky, e.returnedObject, "Invalid response should have been corrected")
                assertTrue(e.constraintViolations.isNotEmpty())
            }
        }

        @Test
        fun `passes correct description of violation to LLM`() {
            // Picky eater
            data class BorderCollie(
                val name: String,
                @field:Pattern(regexp = "^mince$", message = "eats field must be 'mince'")
                val eats: String,
            )

            val invalidHusky = BorderCollie("Husky", eats = "kibble")
            val validHusky = BorderCollie("Husky", eats = "mince")
            val fakeChatModel = FakeChatModel(
                responses = listOf(
                    jacksonObjectMapper().writeValueAsString(invalidHusky),
                    jacksonObjectMapper().writeValueAsString(validHusky),
                )
            )
            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(fakeChatModel)
            val createdDog = setup.llmOperations.createObject(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = BorderCollie::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            val secondPrompt = fakeChatModel.promptsPassed[1].toString()
            assertTrue(secondPrompt.contains("eats field must be 'mince'"), "Prompt mentions validation violation")

            assertEquals(validHusky, createdDog, "Invalid response should have been corrected")
        }

        @Test
        fun `does not pass description of validation rules to LLM if so configured`() {
            // Picky eater
            data class BorderCollie(
                val name: String,
                @field:Pattern(regexp = "^mince$", message = "eats field must be 'mince'")
                val eats: String,
            )

            val validHusky = BorderCollie("Husky", eats = "mince")
            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().writeValueAsString(validHusky)
            )
            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(
                fakeChatModel,
                LlmDataBindingProperties(sendValidationInfo = false)
            )
            val createdDog = setup.llmOperations.createObject(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = BorderCollie::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            val firstPrompt = fakeChatModel.promptsPassed[0].toString()
            assertFalse(firstPrompt.contains("eats field must be 'mince'"), "Prompt mentions validation violation")
            assertEquals(validHusky, createdDog, "Invalid response should have been corrected")
        }

        @Test
        fun `passes correct description of validation rules to LLM if so configured`() {
            // Picky eater
            data class BorderCollie(
                val name: String,
                @field:Pattern(regexp = "^mince$", message = "eats field must be 'mince'")
                val eats: String,
            )

            val validHusky = BorderCollie("Husky", eats = "mince")
            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().writeValueAsString(validHusky)
            )
            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(
                fakeChatModel = fakeChatModel,
                dataBindingProperties = LlmDataBindingProperties(
                    sendValidationInfo = true,
                )
            )
            val createdDog = setup.llmOperations.createObject(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                outputClass = BorderCollie::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            val firstPrompt = fakeChatModel.promptsPassed[0].toString()
            assertTrue(firstPrompt.contains("eats field must be 'mince'"), "Prompt mentions validation violation")

            assertEquals(validHusky, createdDog, "Invalid response should have been corrected")
        }

        @Test
        fun `does not validate if interaction validation is set to false`() {
            // Picky eater
            data class BorderCollie(
                val name: String,
                @field:Pattern(regexp = "^mince$", message = "eats field must be 'mince'")
                val eats: String,
            )

            val invalidHusky = BorderCollie("Husky", eats = "kibble")
            val fakeChatModel = FakeChatModel(
                jacksonObjectMapper().writeValueAsString(invalidHusky)
            )
            val prompt =
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
            val setup = createChatClientLlmOperations(
                fakeChatModel = fakeChatModel,
                dataBindingProperties = LlmDataBindingProperties(
                    sendValidationInfo = true,
                )
            )
            val createdDog = setup.llmOperations.createObject(
                messages = listOf(UserMessage(prompt)),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions(),
                    validation = false
                ),
                outputClass = BorderCollie::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )
            assertEquals(invalidHusky, createdDog, "Invalid response should have been corrected")
        }

        @Test
        fun `field filter suppresses constraint violation for excluded field`() {
            data class BorderCollie(
                val name: String,
                @field:Pattern(regexp = "^mince$", message = "eats field must be 'mince'")
                val eats: String,
            )

            val invalidHusky = BorderCollie("Husky", eats = "kibble")
            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(invalidHusky))
            val setup = createChatClientLlmOperations(fakeChatModel)

            // Exclude 'eats' from the field filter — its constraint violation should be ignored
            val result = setup.llmOperations.createObject(
                messages = listOf(UserMessage("prompt")),
                interaction = LlmInteraction(
                    id = InteractionId("id"),
                    llm = LlmOptions(),
                    fieldFilter = Predicate { field -> field.name != "eats" },
                ),
                outputClass = BorderCollie::class.java,
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            assertEquals(invalidHusky, result, "Filtered-out field violation should not block the result")
        }

        @Test
        fun `field filter does not suppress constraint violation for included field`() {
            data class BorderCollie(
                val name: String,
                @field:Pattern(regexp = "^mince$", message = "eats field must be 'mince'")
                val eats: String,
            )

            val invalidHusky = BorderCollie("Husky", eats = "kibble")
            val fakeChatModel = FakeChatModel(jacksonObjectMapper().writeValueAsString(invalidHusky))
            val setup = createChatClientLlmOperations(fakeChatModel)

            // 'eats' is still included in the filter — violation should be raised
            try {
                setup.llmOperations.createObject(
                    messages = listOf(UserMessage("prompt")),
                    interaction = LlmInteraction(
                        id = InteractionId("id"),
                        llm = LlmOptions(),
                        fieldFilter = Predicate { true },
                    ),
                    outputClass = BorderCollie::class.java,
                    action = SimpleTestAgent.actions.first(),
                    agentProcess = setup.mockAgentProcess,
                )
                fail("Should have thrown InvalidLlmReturnTypeException")
            } catch (e: InvalidLlmReturnTypeException) {
                assertTrue(e.constraintViolations.any { it.propertyPath.toString() == "eats" })
            }
        }
    }

    /**
     * Test of lower level internal implementation
     */
    @Nested
    inner class StreamedChunks {

        @Test
        fun `rawChunksToLines should handle single line chunks`() {
            val chunks = Flux.just("line1\n", "line2\n")

            val setup = createChatClientLlmOperations(FakeChatModel("fake"))
            val result = (setup.streamingLlmOperations as ChatClientLlmOperations).rawChunksToLines(chunks)

            StepVerifier.create(result)
                .expectNext("line1")
                .expectNext("line2")
                .verifyComplete()
        }

        @Test
        fun `rawChunksToLines should handle multi-line chunks from Anthropic`() {
            val chunks = Flux.just(".\n</think>\n\n{\"")

            val setup = createChatClientLlmOperations(FakeChatModel("fake"))
            val result = (setup.streamingLlmOperations as ChatClientLlmOperations).rawChunksToLines(chunks)

            StepVerifier.create(result)
                .expectNext(".")
                .expectNext("</think>")
                .expectNext("{\"")
                .verifyComplete()
        }

        @Test
        fun `rawChunksToLines should handle incomplete lines across chunks`() {
            val chunks = Flux.just("partial", " line\n", "complete\n")

            val setup = createChatClientLlmOperations(FakeChatModel("fake"))
            val result = (setup.streamingLlmOperations as ChatClientLlmOperations).rawChunksToLines(chunks)

            StepVerifier.create(result)
                .expectNext("partial line")
                .expectNext("complete")
                .verifyComplete()
        }

        @Test
        fun `rawChunksToLines should emit final incomplete line`() {
            val chunks = Flux.just("line1\n", "incomplete")

            val setup = createChatClientLlmOperations(FakeChatModel("fake"))
            val result = (setup.streamingLlmOperations as ChatClientLlmOperations).rawChunksToLines(chunks)

            StepVerifier.create(result)
                .expectNext("line1")
                .expectNext("incomplete")
                .verifyComplete()
        }
    }

    /**
     * Tests for useMessageStreamer=true (decoupled streaming path via LlmMessageStreamer).
     */
    @Nested
    inner class MessageStreamerTests {

        @Test
        fun `should use LlmMessageStreamer when useMessageStreamer is true`() {
            // Given
            val fakeChatModel = FakeChatModel(listOf("streamed ", "content"))

            val setup = createChatClientLlmOperations(fakeChatModel, useMessageStreamer = true)

            // When
            val result = setup.streamingLlmOperations.generateStream(
                listOf(UserMessage("test")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            // Then
            StepVerifier.create(result)
                .expectNext("streamed ")
                .expectNext("content")
                .verifyComplete()
        }

        @Test
        fun `should prepend prompt contributions as system message`() {
            // Given
            val fakeChatModel = FakeChatModel(listOf("response"))

            val setup = createChatClientLlmOperations(fakeChatModel, useMessageStreamer = true)

            // When
            val result = setup.streamingLlmOperations.generateStream(
                listOf(UserMessage("user message")),
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            // Then
            StepVerifier.create(result)
                .expectNext("response")
                .verifyComplete()
        }

        @Test
        fun `should handle object streaming`() {
            // Given
            val fakeChatModel = FakeChatModel(listOf("{\"name\":\"Test\",\"value\":42}\n"))

            val setup = createChatClientLlmOperations(fakeChatModel, useMessageStreamer = true)

            // When
            val result = setup.streamingLlmOperations.createObjectStreamWithThinking(
                messages = listOf(UserMessage("test")),
                outputClass = TestItem::class.java,
                interaction = LlmInteraction(
                    id = InteractionId("id"), llm = LlmOptions()
                ),
                action = SimpleTestAgent.actions.first(),
                agentProcess = setup.mockAgentProcess,
            )

            // Then
            StepVerifier.create(result)
                .expectNextMatches {
                    it.isObject() && it.getObject()?.name == "Test" && it.getObject()?.value == 42
                }
                .expectComplete()
                .verify(Duration.ofSeconds(1))
        }
    }
}
