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
package com.embabel.agent.spi.support.springai

import com.embabel.agent.api.common.Asyncer
import com.embabel.agent.api.common.InteractionId
import com.embabel.agent.spi.streaming.StreamingCapabilityDetector
import com.embabel.agent.api.event.LlmRequestEvent
import com.embabel.agent.api.tool.Tool
import com.embabel.agent.api.tool.config.ToolLoopConfiguration
import com.embabel.agent.core.Action
import com.embabel.agent.core.AgentProcess
import com.embabel.agent.core.support.LlmInteraction
import com.embabel.agent.core.support.toEmbabelUsage
import com.embabel.agent.spi.AutoLlmSelectionCriteriaResolver
import com.embabel.agent.spi.LlmService
import com.embabel.agent.spi.ToolDecorator
import com.embabel.agent.spi.loop.AutoCorrectionPolicy
import com.embabel.agent.spi.loop.LlmMessageSender
import com.embabel.agent.spi.loop.ToolLoopFactory
import com.embabel.agent.spi.loop.streaming.LlmMessageStreamer
import com.embabel.agent.spi.support.LlmDataBindingProperties
import com.embabel.agent.spi.support.LlmOperationsPromptsProperties
import com.embabel.agent.spi.support.MaybeReturn
import com.embabel.agent.spi.support.OutputConverter
import com.embabel.agent.spi.support.PROMPT_ELEMENT_SEPARATOR
import com.embabel.agent.spi.support.ToolLoopLlmOperations
import com.embabel.agent.spi.support.ToolResolutionHelper
import com.embabel.agent.spi.support.guardrails.validateAssistantResponse
import com.embabel.agent.spi.support.guardrails.validateUserInput
import com.embabel.agent.spi.support.springai.streaming.SpringAiLlmMessageStreamer
import com.embabel.agent.spi.validation.DefaultValidationPromptGenerator
import com.embabel.agent.spi.validation.ValidationPromptGenerator
import com.embabel.chat.Message
import com.embabel.common.ai.converters.FilteringJacksonOutputConverter
import com.embabel.common.ai.converters.streaming.StreamingJacksonOutputConverter
import com.embabel.common.ai.model.LlmOptions
import com.embabel.common.ai.model.ModelProvider
import com.embabel.common.core.streaming.StreamingEvent
import com.embabel.common.core.thinking.ThinkingException
import com.embabel.common.core.thinking.ThinkingResponse
import com.embabel.common.core.thinking.spi.InternalThinkingApi
import com.embabel.common.core.thinking.spi.extractAllThinkingBlocks
import com.embabel.common.textio.template.TemplateRenderer
import com.fasterxml.jackson.databind.DatabindException
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import io.micrometer.observation.ObservationRegistry
import jakarta.annotation.PostConstruct
import jakarta.validation.Validator
import java.lang.reflect.ParameterizedType
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import javax.annotation.concurrent.ThreadSafe
import org.springframework.ai.chat.client.ChatClient
import org.springframework.ai.chat.client.ChatClientCustomizer
import org.springframework.ai.chat.client.ResponseEntity
import org.springframework.ai.chat.client.advisor.observation.DefaultAdvisorObservationConvention
import org.springframework.ai.chat.client.observation.DefaultChatClientObservationConvention
import org.springframework.ai.chat.messages.SystemMessage
import org.springframework.ai.chat.messages.UserMessage
import org.springframework.ai.chat.model.ChatResponse
import org.springframework.ai.chat.prompt.Prompt
import org.springframework.beans.factory.getBeansOfType
import org.springframework.context.ApplicationContext
import org.springframework.core.ParameterizedTypeReference
import org.springframework.retry.support.RetrySynchronizationManager
import org.springframework.stereotype.Service
import reactor.core.publisher.Flux
import reactor.core.publisher.Mono

// Log message constants to avoid duplication
private const val LLM_TIMEOUT_MESSAGE = "LLM {}: attempt {} timed out after {}ms"
private const val LLM_INTERRUPTED_MESSAGE = "LLM {}: attempt {} was interrupted"

/**
 * Spring AI implementation of LlmOperations using ChatClient.
 *
 * This class extends [ToolLoopLlmOperations] to inherit the framework-agnostic
 * tool loop implementation, and adds Spring AI-specific functionality:
 * - Spring AI converters for structured output
 * - ChatClient for the legacy Spring AI tool handling path
 * - Spring AI message/prompt building
 *
 * @param modelProvider ModelProvider to get the LLM model
 * @param toolDecorator ToolDecorator to decorate tools to make them aware of platform
 * @param templateRenderer TemplateRenderer to render templates
 * @param dataBindingProperties properties
 * @param useMessageStreamer When true, delegates raw streaming to [LlmMessageStreamer] instead of calling Spring AI ChatClient directly. This decouples streaming from Spring AI, enabling vendor-neutral implementations.
 */
@ThreadSafe
@Service
internal class ChatClientLlmOperations(
    modelProvider: ModelProvider,
    toolDecorator: ToolDecorator,
    validator: Validator,
    validationPromptGenerator: ValidationPromptGenerator = DefaultValidationPromptGenerator(),
    templateRenderer: TemplateRenderer,
    dataBindingProperties: LlmDataBindingProperties = LlmDataBindingProperties(),
    llmOperationsPromptsProperties: LlmOperationsPromptsProperties = LlmOperationsPromptsProperties(),
    private val applicationContext: ApplicationContext? = null,
    autoLlmSelectionCriteriaResolver: AutoLlmSelectionCriteriaResolver = AutoLlmSelectionCriteriaResolver.DEFAULT,
    objectMapper: ObjectMapper = jacksonObjectMapper().registerModule(JavaTimeModule()),
    observationRegistry: ObservationRegistry = ObservationRegistry.NOOP,
    private val customizers: List<ChatClientCustomizer> = emptyList(),
    asyncer: Asyncer,
    toolLoopFactory: ToolLoopFactory = ToolLoopFactory.create(ToolLoopConfiguration(), asyncer, AutoCorrectionPolicy()),
    private val useMessageStreamer: Boolean = false,
) : ToolLoopLlmOperations(
    toolDecorator = toolDecorator,
    modelProvider = modelProvider,
    validator = validator,
    validationPromptGenerator = validationPromptGenerator,
    dataBindingProperties = dataBindingProperties,
    autoLlmSelectionCriteriaResolver = autoLlmSelectionCriteriaResolver,
    promptsProperties = llmOperationsPromptsProperties,
    objectMapper = objectMapper,
    observationRegistry = observationRegistry,
    toolLoopFactory = toolLoopFactory,
    asyncer = asyncer,
    templateRenderer = templateRenderer,
), LlmOperationsIncludingStreaming {

    @PostConstruct
    private fun logPropertyConfiguration() {
        val dataBindingFromContext = applicationContext?.runCatching {
            getBeansOfType<LlmDataBindingProperties>().values.firstOrNull()
        }?.getOrNull()

        val promptsFromContext = applicationContext?.runCatching {
            getBeansOfType<LlmOperationsPromptsProperties>().values.firstOrNull()
        }?.getOrNull()

        if (dataBindingFromContext === dataBindingProperties) {
            logger.debug("LLM Data Binding: Using Spring-managed properties")
        } else {
            logger.warn("LLM Data Binding: Using fallback defaults")
        }

        if (promptsFromContext === promptsProperties) {
            logger.debug("LLM Prompts: Using Spring-managed properties")
        } else {
            logger.warn("LLM Prompts: Using fallback defaults")
        }

        logger.info(
            "Current LLM settings: maxAttempts={}, fixedBackoffMillis={}ms, timeout={}s",
            dataBindingProperties.maxAttempts,
            dataBindingProperties.fixedBackoffMillis,
            promptsProperties.defaultTimeout.seconds,
        )
    }

    // ====================================
    // OVERRIDES FROM ToolLoopLlmOperations
    // ====================================

    override fun createMessageSender(
        llm: LlmService<*>,
        options: LlmOptions,
        llmRequestEvent: LlmRequestEvent<*>?,
    ): LlmMessageSender {
        if (llmRequestEvent != null) {
            val springAiLlm = requireSpringAiLlm(llm)
            val chatOptions = springAiLlm.optionsConverter.convertOptions(options)
            val instrumentedModel = InstrumentedChatModel(springAiLlm.chatModel, llmRequestEvent)
            return SpringAiLlmMessageSender(instrumentedModel, chatOptions, springAiLlm.toolResponseContentAdapter)
        }
        return llm.createMessageSender(options)
    }

    override fun <O> createOutputConverter(
        outputClass: Class<O>,
        interaction: LlmInteraction,
    ): OutputConverter<O> {
        val springAiConverter = ExceptionWrappingConverter(
            expectedType = outputClass,
            delegate = WithExampleConverter(
                delegate = SuppressThinkingConverter(
                    FilteringJacksonOutputConverter(
                        clazz = outputClass,
                        objectMapper = objectMapper,
                        fieldFilter = interaction.fieldFilter,
                    )
                ),
                outputClass = outputClass,
                ifPossible = false,
                generateExamples = shouldGenerateExamples(interaction),
            )
        )
        return SpringAiOutputConverterAdapter(springAiConverter)
    }

    override fun sanitizeStringOutput(text: String): String {
        return stringWithoutThinkBlocks(text)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <O> createMaybeReturnOutputConverter(
        outputClass: Class<O>,
        interaction: LlmInteraction,
    ): OutputConverter<MaybeReturn<O>> {
        val typeReference = createParameterizedTypeReference<MaybeReturn<*>>(
            MaybeReturn::class.java,
            outputClass,
        )
        val springAiConverter = ExceptionWrappingConverter(
            expectedType = MaybeReturn::class.java,
            delegate = WithExampleConverter(
                delegate = SuppressThinkingConverter(
                    FilteringJacksonOutputConverter(
                        typeReference = typeReference,
                        objectMapper = objectMapper,
                        fieldFilter = interaction.fieldFilter,
                    )
                ),
                outputClass = outputClass as Class<MaybeReturn<*>>,
                ifPossible = true,
                generateExamples = shouldGenerateExamples(interaction),
            )
        )
        return SpringAiOutputConverterAdapter(springAiConverter) as OutputConverter<MaybeReturn<O>>
    }

    // emitCallEvent is intentionally not overridden here.
    // InstrumentedChatModel emits ChatModelCallEvent at the actual ChatModel.call() point,
    // capturing the fully augmented prompt (with resolved options, tool schemas, etc.)
    // and firing once per tool-loop iteration rather than once before the loop.

    /**
     * Extracts a non-null [ChatResponse] from the call response, throwing a descriptive
     * [IllegalStateException] instead of an NPE when the response is null.
     * A null response typically indicates an invalid API key or insufficient model permissions.
     */
    private fun requireChatResponse(
        callResponse: ChatClient.CallResponseSpec,
        interaction: LlmInteraction,
    ): ChatResponse {
        return callResponse.chatResponse()
            ?: throw IllegalStateException(
                "LLM call for interaction '${interaction.id.value}' returned no response. " +
                        "This typically indicates an invalid API key or insufficient permissions for the configured model."
            )
    }

    /**
     * Extracts a non-null entity from the response, throwing a descriptive
     * [IllegalStateException] instead of an NPE when the entity is null.
     */
    private fun <T> requireEntity(
        responseEntity: ResponseEntity<ChatResponse, T>,
        interaction: LlmInteraction,
    ): T {
        return responseEntity.entity
            ?: throw IllegalStateException(
                "LLM call for interaction '${interaction.id.value}' returned no parseable entity. " +
                        "This typically indicates an invalid API key or insufficient permissions for the configured model."
            )
    }

    /**
     * ******************************************
     * LEGACY SPRING AI IMPLEMENTATION
     * ******************************************
     * Transform messages to an object with thinking block extraction.
     * Uses Spring AI's internal tool loop.
     */
    @OptIn(InternalThinkingApi::class)
    internal fun <O> doTransformWithThinkingSpringAi(
        messages: List<Message>,
        interaction: LlmInteraction,
        outputClass: Class<O>,
        llmRequestEvent: LlmRequestEvent<O>?,
        agentProcess: AgentProcess?,
        action: Action?,
    ): ThinkingResponse<O> {
        logger.debug("LLM transform for interaction {} with thinking extraction", interaction.id.value)

        val llm = chooseLlm(interaction.llm)
        val chatClient = createChatClient(llm, llmRequestEvent)
        val promptContributions = buildPromptContributions(interaction, llm)

        // Create converter chain once for both schema format and actual conversion
        val converter = if (outputClass != String::class.java) {
            ExceptionWrappingConverter(
                expectedType = outputClass,
                delegate = WithExampleConverter(
                    delegate = SuppressThinkingConverter(
                        FilteringJacksonOutputConverter(
                            clazz = outputClass,
                            objectMapper = objectMapper,
                            fieldFilter = interaction.fieldFilter,
                        )
                    ),
                    outputClass = outputClass,
                    ifPossible = false,
                    generateExamples = shouldGenerateExamples(interaction),
                )
            )
        } else null

        val schemaFormat = converter?.getFormat()

        val springAiPrompt = if (schemaFormat != null) {
            buildPromptWithSchema(promptContributions, messages, schemaFormat)
        } else {
            buildBasicPrompt(promptContributions, messages)
        }

        // Guardrails: Pre-validation of user input
        val userMessages = messages.filterIsInstance<com.embabel.chat.UserMessage>()
        validateUserInput(userMessages, interaction, llmRequestEvent?.agentProcess?.blackboard)

        val chatOptions = requireSpringAiLlm(llm).optionsConverter.convertOptions(interaction.llm)
        val timeoutMillis = getTimeoutMillis(interaction.llm)

        // Resolve tool groups and decorate tools
        val tools = resolveAndDecorateTools(interaction, agentProcess, action)

        return dataBindingProperties.retryTemplate(interaction.id.value)
            .execute<ThinkingResponse<O>, DatabindException> {
                val attempt = (RetrySynchronizationManager.getContext()?.retryCount ?: 0) + 1

                val future = asyncer.async {
                    chatClient
                        .prompt(springAiPrompt)
                        .toolCallbacks(tools.toSpringToolCallbacks())
                        .options(chatOptions)
                        .call()
                }

                val callResponse = try {
                    future.get(
                        timeoutMillis,
                        TimeUnit.MILLISECONDS
                    ) // NOSONAR: CompletableFuture.get() is not collection access
                } catch (e: Exception) {
                    handleFutureException(e, future, interaction, timeoutMillis, attempt)
                }

                logger.debug("LLM call completed for interaction {}", interaction.id.value)

                // Convert response with thinking extraction using manual converter chains
                if (outputClass == String::class.java) {
                    val chatResponse = requireChatResponse(callResponse, interaction)
                    recordUsage(llm, chatResponse, llmRequestEvent)
                    val rawText = chatResponse.result.output.text as String

                    val thinkingBlocks = extractAllThinkingBlocks(rawText)
                    logger.debug("Extracted {} thinking blocks for String response", thinkingBlocks.size)

                    val thinkingResponse = ThinkingResponse(
                        result = rawText as O, // NOSONAR: Safe cast verified by outputClass == String::class.java check
                        thinkingBlocks = thinkingBlocks
                    )

                    // Guardrails: Post-validation of assistant response (ThinkingResponse with thinking blocks)
                    validateAssistantResponse(thinkingResponse, interaction, llmRequestEvent?.agentProcess?.blackboard)

                    thinkingResponse
                } else {
                    // Extract thinking blocks from raw response text FIRST
                    val chatResponse = requireChatResponse(callResponse, interaction)
                    recordUsage(llm, chatResponse, llmRequestEvent)
                    val rawText = chatResponse.result.output.text ?: ""

                    val thinkingBlocks = extractAllThinkingBlocks(rawText)
                    logger.debug(
                        "Extracted {} thinking blocks for {} response",
                        thinkingBlocks.size,
                        outputClass.simpleName
                    )

                    // Execute converter chain manually instead of using responseEntity
                    try {
                        val result = converter!!.convert(rawText)

                        val thinkingResponse: ThinkingResponse<O> = ThinkingResponse(
                            result = result!! as O,
                            thinkingBlocks = thinkingBlocks
                        )

                        // Guardrails: Post-validation of assistant response (ThinkingResponse with thinking blocks)
                        validateAssistantResponse(
                            thinkingResponse,
                            interaction,
                            llmRequestEvent?.agentProcess?.blackboard
                        )

                        thinkingResponse
                    } catch (e: Exception) {
                        // Preserve thinking blocks in exceptions
                        throw ThinkingException(
                            message = "Conversion failed: ${e.message}",
                            thinkingBlocks = thinkingBlocks
                        )
                    }
                }
            }
    }

    /**
     * Spring AI implementation of transform with thinking extraction using IfPossible pattern.
     */
    @OptIn(InternalThinkingApi::class)
    internal fun <O> doTransformWithThinkingIfPossibleSpringAi(
        messages: List<Message>,
        interaction: LlmInteraction,
        outputClass: Class<O>,
        llmRequestEvent: LlmRequestEvent<O>?,
        agentProcess: AgentProcess?,
        action: Action?,
    ): Result<ThinkingResponse<O>> {
        return try {
            val maybeReturnPromptContribution = templateRenderer.renderLoadedTemplate(
                promptsProperties.maybePromptTemplate,
                emptyMap(),
            )

            val llm = chooseLlm(interaction.llm)
            val chatClient = createChatClient(llm, llmRequestEvent)
            val promptContributions = buildPromptContributions(interaction, llm)

            val typeReference = createParameterizedTypeReference<MaybeReturn<*>>(
                MaybeReturn::class.java,
                outputClass,
            )

            // Create converter chain BEFORE LLM call to get schema format
            val converter = ExceptionWrappingConverter(
                expectedType = MaybeReturn::class.java,
                delegate = WithExampleConverter(
                    delegate = SuppressThinkingConverter(
                        FilteringJacksonOutputConverter(
                            typeReference = typeReference,
                            objectMapper = objectMapper,
                            fieldFilter = interaction.fieldFilter,
                        )
                    ),
                    outputClass = outputClass as Class<MaybeReturn<*>>, // NOSONAR: Safe cast for MaybeReturn wrapper pattern
                    ifPossible = true,
                    generateExamples = shouldGenerateExamples(interaction),
                )
            )

            // Get the complete format (examples + JSON schema)
            val schemaFormat = converter.getFormat()

            val springAiPrompt = buildPromptWithMaybeReturnAndSchema(
                promptContributions,
                messages,
                maybeReturnPromptContribution,
                schemaFormat
            )

            // Guardrails: Pre-validation of user input
            val userMessages = messages.filterIsInstance<com.embabel.chat.UserMessage>()
            validateUserInput(userMessages, interaction, llmRequestEvent?.agentProcess?.blackboard)

            val chatOptions = requireSpringAiLlm(llm).optionsConverter.convertOptions(interaction.llm)
            val timeoutMillis = getTimeoutMillis(interaction.llm)

            // Resolve tool groups and decorate tools
            val tools = resolveAndDecorateTools(interaction, agentProcess, action)

            val result = dataBindingProperties.retryTemplate(interaction.id.value)
                .execute<Result<ThinkingResponse<O>>, DatabindException> {
                    val future = asyncer.async {
                        chatClient
                            .prompt(springAiPrompt)
                            .toolCallbacks(tools.toSpringToolCallbacks())
                            .options(chatOptions)
                            .call()
                    }

                    val callResponse = try {
                        future.get(
                            timeoutMillis,
                            TimeUnit.MILLISECONDS
                        ) // NOSONAR: CompletableFuture.get() is not collection access
                    } catch (e: Exception) {
                        val attempt = (RetrySynchronizationManager.getContext()?.retryCount ?: 0) + 1
                        return@execute handleFutureExceptionAsResult(e, future, interaction, timeoutMillis, attempt)
                    }

                    // Extract thinking blocks from raw text FIRST
                    val chatResponse = requireChatResponse(callResponse, interaction)
                    recordUsage(llm, chatResponse, llmRequestEvent)
                    val rawText = chatResponse.result.output.text ?: ""
                    val thinkingBlocks = extractAllThinkingBlocks(rawText)

                    // Execute converter chain manually instead of using responseEntity
                    try {
                        val maybeResult = converter.convert(rawText)

                        // Convert MaybeReturn<O> to Result<ThinkingResponse<O>> with extracted thinking blocks
                        val result =
                            maybeResult!!.toResult() as Result<O> // NOSONAR: Safe cast, MaybeReturn<O>.toResult() returns Result<O>
                        when {
                            result.isSuccess -> {
                                val thinkingResponse = ThinkingResponse(
                                    result = result.getOrThrow(),
                                    thinkingBlocks = thinkingBlocks
                                )

                                // Guardrails: Post-validation of assistant response (ThinkingResponse with thinking blocks)
                                validateAssistantResponse(
                                    thinkingResponse,
                                    interaction,
                                    llmRequestEvent?.agentProcess?.blackboard
                                )

                                Result.success(thinkingResponse)
                            }

                            else -> {
                                // Validate thinking blocks even when object creation fails
                                if (thinkingBlocks.isNotEmpty()) {
                                    val thinkingResponse = ThinkingResponse(
                                        result = null,
                                        thinkingBlocks = thinkingBlocks
                                    )
                                    validateAssistantResponse(
                                        thinkingResponse,
                                        interaction,
                                        llmRequestEvent?.agentProcess?.blackboard
                                    )
                                }

                                Result.failure(
                                    ThinkingException(
                                        message = "Object creation not possible: ${result.exceptionOrNull()?.message ?: "Unknown error"}",
                                        thinkingBlocks = thinkingBlocks
                                    )
                                )
                            }
                        }
                    } catch (e: Exception) {
                        // Other failures, preserve thinking blocks
                        Result.failure(
                            ThinkingException(
                                message = "Conversion failed: ${e.message}",
                                thinkingBlocks = thinkingBlocks
                            )
                        )
                    }
                }
            result
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // ====================================
    // SPRING AI SPECIFIC UTILITIES
    // ====================================

    @Suppress("UNCHECKED_CAST")
    private fun <T> createParameterizedTypeReference(
        rawType: Class<*>,
        typeArgument: Class<*>,
    ): ParameterizedTypeReference<T> {
        // Create a type with proper generic information
        val type = object : ParameterizedType {
            override fun getRawType() = rawType
            override fun getActualTypeArguments() = arrayOf(typeArgument)
            override fun getOwnerType() = null
        }

        // Create a ParameterizedTypeReference that uses our custom type
        return object : ParameterizedTypeReference<T>() {
            override fun getType() = type
        }
    }

    private fun getLlm(interaction: LlmInteraction): LlmService<*> = chooseLlm(interaction.llm)

    /**
     * Require the LLM to be a SpringAiLlm for Spring AI specific operations.
     */
    private fun requireSpringAiLlm(llm: LlmService<*>): SpringAiLlmService {
        return llm as? SpringAiLlmService
            ?: throw IllegalStateException("ChatClientLlmOperations requires SpringAiLlm, got ${llm::class.simpleName}")
    }

    /**
     * Create a chat client for the given LlmService.
     * Requires the LlmService to be a SpringAiLlm.
     *
     * When [llmRequestEvent] is provided, the underlying [ChatModel] is wrapped in an
     * [InstrumentedChatModel] that emits a [ChatModelCallEvent] with the **final augmented
     * prompt** (including tool schemas, format instructions, etc.) at the point Spring AI
     * actually calls the model. This replaces the need for manual event emission at each
     * call site — the decorator handles it transparently.
     *
     * @param llm the LLM service to create a client for
     * @param llmRequestEvent optional domain context; when present, enables instrumentation
     */
    private fun createChatClient(
        llm: LlmService<*>,
        llmRequestEvent: LlmRequestEvent<*>? = null,
    ): ChatClient {
        val springAiLlm = requireSpringAiLlm(llm)
        val chatModel = if (llmRequestEvent != null) {
            InstrumentedChatModel(springAiLlm.chatModel, llmRequestEvent)
        } else {
            springAiLlm.chatModel
        }
        return ChatClient
            .builder(
                chatModel,
                observationRegistry,
                DefaultChatClientObservationConvention(),
                DefaultAdvisorObservationConvention()
            ).also { builder ->
                customizers.forEach {
                    it.customize(builder)
                }
            }.build()
    }

    private fun recordUsage(
        llm: LlmService<*>,
        chatResponse: ChatResponse,
        llmRequestEvent: LlmRequestEvent<*>?,
    ) {
        logger.debug("Usage is {}", chatResponse.metadata.usage)
        recordUsage(llm, chatResponse.metadata.usage.toEmbabelUsage(), llmRequestEvent)
    }

    // ====================================
    // SPRING AI PROMPT BUILDERS
    // ====================================

    /**
     * Base prompt builder - consolidates all system messages at the beginning.
     * Extracts SystemMessages from the input messages and merges with promptContributions
     * to ensure proper message ordering for cross-model compatibility.
     */
    private fun buildBasicPrompt(
        promptContributions: String,
        messages: List<Message>,
    ): Prompt {
        val (systemMessages, nonSystemMessages) = partitionMessages(messages)
        val allSystemContent = buildConsolidatedSystemMessage(promptContributions, *systemMessages.toTypedArray())
        return Prompt(
            buildList {
                if (allSystemContent.isNotEmpty()) {
                    add(SystemMessage(allSystemContent))
                }
                addAll(nonSystemMessages.map { it.toSpringAiMessage() })
            }
        )
    }

    /**
     * Extends basic prompt with maybeReturn user message.
     * Consolidates all system messages at the beginning for proper ordering.
     */
    private fun buildPromptWithMaybeReturn(
        promptContributions: String,
        messages: List<Message>,
        maybeReturnPrompt: String,
    ): Prompt {
        val (systemMessages, nonSystemMessages) = partitionMessages(messages)
        val allSystemContent = buildConsolidatedSystemMessage(promptContributions, *systemMessages.toTypedArray())
        return Prompt(
            buildList {
                if (allSystemContent.isNotEmpty()) {
                    add(SystemMessage(allSystemContent))
                }
                add(UserMessage(maybeReturnPrompt))
                addAll(nonSystemMessages.map { it.toSpringAiMessage() })
            }
        )
    }

    /**
     * Builds a prompt with schema format consolidated into the system message.
     * All system content is placed at the beginning to ensure proper message ordering
     * for cross-model compatibility (OpenAI best practice, required by DeepSeek, etc.).
     */
    private fun buildPromptWithSchema(
        promptContributions: String,
        messages: List<Message>,
        schemaFormat: String,
    ): Prompt {
        logger.debug("Injected schema format for thinking extraction: {}", schemaFormat)
        val (systemMessages, nonSystemMessages) = partitionMessages(messages)
        val allSystemContent = buildConsolidatedSystemMessage(
            promptContributions,
            *systemMessages.toTypedArray(),
            schemaFormat
        )
        return Prompt(
            buildList {
                if (allSystemContent.isNotEmpty()) {
                    add(SystemMessage(allSystemContent))
                }
                addAll(nonSystemMessages.map { it.toSpringAiMessage() })
            }
        )
    }

    /**
     * Combines maybeReturn user message with schema format.
     * All system content is consolidated at the beginning for proper message ordering.
     */
    private fun buildPromptWithMaybeReturnAndSchema(
        promptContributions: String,
        messages: List<Message>,
        maybeReturnPrompt: String,
        schemaFormat: String,
    ): Prompt {
        val (systemMessages, nonSystemMessages) = partitionMessages(messages)
        val allSystemContent = buildConsolidatedSystemMessage(
            promptContributions,
            *systemMessages.toTypedArray(),
            schemaFormat
        )
        return Prompt(
            buildList {
                if (allSystemContent.isNotEmpty()) {
                    add(SystemMessage(allSystemContent))
                }
                add(UserMessage(maybeReturnPrompt))
                addAll(nonSystemMessages.map { it.toSpringAiMessage() })
            }
        )
    }

    /**
     * Partitions messages into system messages (content only) and non-system messages.
     * This enables consolidating all system content at the beginning of the prompt.
     */
    private fun partitionMessages(messages: List<Message>): Pair<List<String>, List<Message>> {
        val systemContent = mutableListOf<String>()
        val nonSystemMessages = mutableListOf<Message>()
        for (message in messages) {
            if (message is com.embabel.chat.SystemMessage) {
                systemContent.add(message.content)
            } else {
                nonSystemMessages.add(message)
            }
        }
        return systemContent to nonSystemMessages
    }

    /**
     * Consolidates multiple system content strings into a single system message.
     * This ensures a single system message at the beginning of the conversation,
     * following OpenAI best practices and ensuring compatibility with models like DeepSeek
     * that have strict message ordering requirements.
     */
    private fun buildConsolidatedSystemMessage(vararg contents: String): String =
        contents.filter { it.isNotEmpty() }.joinToString("\n\n")

    // ====================================
    // EXCEPTION HANDLING
    // ====================================

    /**
     * Handles exceptions from CompletableFuture execution during LLM calls.
     *
     * Provides centralized exception handling for timeout, interruption, and execution failures.
     * Cancels the future, logs appropriate warnings/errors, and throws descriptive RuntimeExceptions.
     *
     * @param e The exception that occurred during future execution
     * @param future The CompletableFuture to cancel on error
     * @param interaction The LLM interaction context for error messages
     * @param timeoutMillis The timeout value for error reporting
     * @param attempt The retry attempt number for logging
     * @throws RuntimeException Always throws with appropriate error message based on exception type
     */
    private fun handleFutureException(
        e: Exception,
        future: CompletableFuture<*>,
        interaction: LlmInteraction,
        timeoutMillis: Long,
        attempt: Int
    ): Nothing {
        when (e) {
            is TimeoutException -> {
                future.cancel(true)
                logger.warn(LLM_TIMEOUT_MESSAGE, interaction.id.value, attempt, timeoutMillis)
                throw RuntimeException(
                    "ChatClient call for interaction ${interaction.id.value} timed out after ${timeoutMillis}ms",
                    e
                )
            }

            is InterruptedException -> {
                future.cancel(true)
                Thread.currentThread().interrupt()
                logger.warn(LLM_INTERRUPTED_MESSAGE, interaction.id.value, attempt)
                throw RuntimeException("ChatClient call for interaction ${interaction.id.value} was interrupted", e)
            }

            is ExecutionException -> {
                future.cancel(true)
                logger.error(
                    "LLM {}: attempt {} failed with execution exception",
                    interaction.id.value,
                    attempt,
                    e.cause
                )
                when (val cause = e.cause) {
                    is RuntimeException -> throw cause
                    is Exception -> throw RuntimeException(
                        "ChatClient call for interaction ${interaction.id.value} failed",
                        cause
                    )

                    else -> throw RuntimeException(
                        "ChatClient call for interaction ${interaction.id.value} failed with unknown error",
                        e
                    )
                }
            }

            else -> throw e
        }
    }

    /**
     * Handles exceptions from CompletableFuture execution, returning Result.failure.
     */
    private fun <O> handleFutureExceptionAsResult(
        e: Exception,
        future: CompletableFuture<*>,
        interaction: LlmInteraction,
        timeoutMillis: Long,
        attempt: Int
    ): Result<ThinkingResponse<O>> {
        return when (e) {
            is TimeoutException -> {
                future.cancel(true)
                logger.warn(LLM_TIMEOUT_MESSAGE, interaction.id.value, attempt, timeoutMillis)
                Result.failure(
                    ThinkingException(
                        message = "ChatClient call for interaction ${interaction.id.value} timed out after ${timeoutMillis}ms",
                        thinkingBlocks = emptyList()
                    )
                )
            }

            is InterruptedException -> {
                future.cancel(true)
                Thread.currentThread().interrupt()
                logger.warn(LLM_INTERRUPTED_MESSAGE, interaction.id.value, attempt)
                Result.failure(
                    ThinkingException(
                        message = "ChatClient call for interaction ${interaction.id.value} was interrupted",
                        thinkingBlocks = emptyList() // No response = no thinking blocks
                    )
                )
            }

            else -> {
                future.cancel(true)
                logger.error("LLM {}: attempt {} failed", interaction.id.value, attempt, e)
                Result.failure(
                    ThinkingException(
                        message = "ChatClient call for interaction ${interaction.id.value} failed: ${e.message}",
                        thinkingBlocks = emptyList() // No response = no thinking blocks
                    )
                )
            }
        }
    }

    // ====================================
    // TOOL RESOLUTION
    // ====================================

    /**
     * Resolves ToolGroups and decorates all tools for streaming operations.
     * Convenience wrapper around [ToolResolutionHelper.resolveAndDecorate].
     * When agentProcess is null, returns interaction.tools without decoration.
     */
    internal fun resolveAndDecorateTools(
        interaction: LlmInteraction,
        agentProcess: AgentProcess?,
        action: Action?,
    ): List<Tool> = ToolResolutionHelper.resolveAndDecorate(
        interaction = interaction,
        agentProcess = agentProcess,
        action = action,
        toolDecorator = toolDecorator,
    )

    override fun supportsStreaming(llmOptions: LlmOptions): Boolean {
        val llm = getLlm(
            LlmInteraction(
                id = InteractionId("capability-check"),
                llm = llmOptions
            )
        )
        val springAiLlm = llm as? SpringAiLlmService ?: return false
        return StreamingCapabilityDetector.supportsStreaming(springAiLlm.chatModel)
    }

    override fun doTransformStream(
        messages: List<Message>,
        interaction: LlmInteraction,
        llmRequestEvent: LlmRequestEvent<String>?,
        agentProcess: AgentProcess?,
        action: Action?,
    ): Flux<String> {
        // Use ChatClientLlmOperations to get LLM and create ChatClient
        val llm = getLlm(interaction)
        val chatClient = createChatClient(llm)

        // Build prompt using helper methods
        val promptContributions = buildPromptContributions(interaction, llm)
        val springAiPrompt = buildBasicPrompt(promptContributions, messages)

        // Guardrails: Pre-validation of user input
        val userMessages = messages.filterIsInstance<com.embabel.chat.UserMessage>()
        validateUserInput(userMessages, interaction, llmRequestEvent?.agentProcess?.blackboard)

        val chatOptions = requireSpringAiLlm(llm).optionsConverter.convertOptions(interaction.llm)

        // Resolve tool groups and decorate tools
        val tools = resolveAndDecorateTools(interaction, agentProcess, action)

        return createStreamInternal(
            chatClient = chatClient,
            messages = messages,
            promptContributions = promptContributions,
            tools = tools,
            chatOptions = chatOptions,
            springAiPrompt = springAiPrompt,
        )
    }

    /**
     * Creates a stream of typed objects from LLM JSONL responses, with thinking content suppressed.
     *
     * This method provides a clean object-only stream by filtering the internal unified stream
     * to exclude thinking content and extract only typed objects.
     *
     * **Stream Characteristics:**
     * - **Input**: Raw LLM chunks containing JSONL + thinking content
     * - **Processing**: Chunks → Lines → Events → Objects (thinking filtered out)
     * - **Output**: `Flux<O>` containing only parsed typed objects
     * - **Error Handling**: Malformed JSON is skipped; stream continues
     * - **Backpressure**: Supports standard Flux operators and subscription patterns
     *
     * **Example Usage:**
     * ```kotlin
     * val objectStream: Flux<User> = doTransformObjectStream(messages, interaction, User::class.java, null)
     *
     * objectStream
     *     .doOnNext { user -> println("Received user: ${user.name}") }
     *     .doOnError { error -> logger.error("Stream error", error) }
     *     .doOnComplete { println("Stream completed") }
     *     .subscribe()
     * ```
     *
     * **Difference from doTransformObjectStreamWithThinking:**
     * - This method: Returns `Flux<O>` with only objects (thinking suppressed)
     * - WithThinking: Returns `Flux<StreamingEvent<O>>` with both thinking and objects
     *
     * @param messages The conversation messages to send to LLM
     * @param interaction LLM configuration and context
     * @param outputClass The target class for object deserialization
     * @param llmRequestEvent Optional event for tracking/observability
     * @return Flux of typed objects, thinking content filtered out
     */
    override fun <O> doTransformObjectStream(
        messages: List<Message>,
        interaction: LlmInteraction,
        outputClass: Class<O>,
        llmRequestEvent: LlmRequestEvent<O>?,
        agentProcess: AgentProcess?,
        action: Action?,
    ): Flux<O> {
        return doTransformObjectStreamInternal(
            messages = messages,
            interaction = interaction,
            outputClass = outputClass,
            llmRequestEvent = llmRequestEvent,
            agentProcess = agentProcess,
            action = action,
        )
            .filter { it.isObject() }
            .map { (it as StreamingEvent.Object).item }
    }

    /**
     * Creates a mixed stream containing both LLM thinking content and typed objects.
     *
     * This method returns the full unified stream without filtering, allowing users to receive
     * both thinking events (LLM reasoning) and object events (parsed JSON data) in the order
     * they appear in the LLM response.
     *
     * **Stream Characteristics:**
     * - **Input**: Raw LLM chunks containing JSONL + thinking content
     * - **Processing**: Chunks → Lines → Events (both thinking and objects preserved)
     * - **Output**: `Flux<StreamingEvent<O>>` with mixed content
     * - **Event Types**: `StreamingEvent.Thinking(content)` and `StreamingEvent.Object(data)`
     * - **Error Handling**: Malformed JSON treated as thinking content; stream continues
     *
     * **Example Usage:**
     * ```kotlin
     * val mixedStream: Flux<StreamingEvent<User>> = doTransformObjectStreamWithThinking(...)
     *
     * mixedStream.subscribe { event ->
     *     when {
     *         event.isThinking() -> println("LLM thinking: ${event.getThinking()}")
     *         event.isObject() -> println("User object: ${event.getObject()}")
     *     }
     * }
     * ```
     *
     * **User Filtering Options:**
     * ```kotlin
     * // Get only thinking content:
     * val thinkingOnly = mixedStream.filter { it.isThinking() }.map { it.getThinking()!! }
     *
     * // Get only objects (equivalent to doTransformObjectStream):
     * val objectsOnly = mixedStream.filter { it.isObject() }.map { it.getObject()!! }
     * ```
     *
     * @param messages The conversation messages to send to LLM
     * @param interaction LLM configuration and context
     * @param outputClass The target class for object deserialization
     * @param llmRequestEvent Optional event for tracking/observability
     * @return Flux of StreamingEvent<O> containing both thinking and object events
     */
    override fun <O> doTransformObjectStreamWithThinking(
        messages: List<Message>,
        interaction: LlmInteraction,
        outputClass: Class<O>,
        llmRequestEvent: LlmRequestEvent<O>?,
        agentProcess: AgentProcess?,
        action: Action?,
    ): Flux<StreamingEvent<O>> {
        return doTransformObjectStreamInternal(
            messages = messages,
            interaction = interaction,
            outputClass = outputClass,
            llmRequestEvent = llmRequestEvent,
            agentProcess = agentProcess,
            action = action,
        )
    }

    /**
     * Internal unified streaming implementation - workhorse -that handles the complete transformation pipeline.
     *
     * This method implements a robust 3-step transformation pipeline:
     * 1. **Raw LLM Chunks**: Receives arbitrary-sized chunks from LLM via Spring AI ChatClient
     * 2. **Line Buffering**: Accumulates chunks into complete logical lines using stateful LineBuffer
     * 3. **Event Generation**: Classifies lines as thinking vs objects, converts to StreamingEvent<O>
     *
     * **Design Principles:**
     * - **Single Source of Truth**: All streaming logic centralized here
     * - **Error Isolation**: Malformed lines don't break the entire stream
     * - **Order Preservation**: Events maintain LLM response order via concatMap
     * - **Backpressure Support**: Full Flux lifecycle support with reactive operators
     *
     * **Event Types Generated:**
     * - `StreamingEvent.Thinking(content)`: LLM reasoning text (from `<think>` blocks or prefix thinking)
     * - `StreamingEvent.Object(data)`: Parsed typed objects from JSONL content
     *
     * **Error Handling Strategy:**
     * - Chunk processing errors: Skip chunk, continue stream
     * - Line classification errors: Treat as thinking content
     * - JSON parsing errors: Skip line, continue processing
     * - Stream continues on individual failures to maximize data recovery
     *
     * **Performance Characteristics:**
     * - Streaming-friendly: no blocking operations
     *
     * @return Unified Flux<StreamingEvent<O>> that public methods can filter as needed
     */
    private fun <O> doTransformObjectStreamInternal(
        messages: List<Message>,
        interaction: LlmInteraction,
        outputClass: Class<O>,
        @Suppress("UNUSED_PARAMETER")
        llmRequestEvent: LlmRequestEvent<O>?,
        agentProcess: AgentProcess?,
        action: Action?,
    ): Flux<StreamingEvent<O>> {
        // Common setup - delegate to ChatClientLlmOperations for LLM setup
        val llm = getLlm(interaction)
        // Chat Client
        val chatClient = createChatClient(llm)
        // Chat Options, additional potential option "streaming"
        val chatOptions = requireSpringAiLlm(llm).optionsConverter.convertOptions(interaction.llm)

        val streamingConverter = StreamingJacksonOutputConverter(
            clazz = outputClass,
            objectMapper = objectMapper,
            fieldFilter = interaction.fieldFilter
        )

        // Build prompt using helper methods, including streaming format instructions
        val promptContributions = buildPromptContributions(interaction, llm)
        val streamingFormatInstructions = streamingConverter.getFormat()
        logger.debug("STREAMING FORMAT INSTRUCTIONS: $streamingFormatInstructions")
        val fullPromptContributions = if (promptContributions.isNotEmpty()) {
            "$promptContributions$PROMPT_ELEMENT_SEPARATOR$streamingFormatInstructions"
        } else {
            streamingFormatInstructions
        }
        val springAiPrompt = buildBasicPrompt(fullPromptContributions, messages)

        // Guardrails: Pre-validation of user input
        val userMessages = messages.filterIsInstance<com.embabel.chat.UserMessage>()
        validateUserInput(userMessages, interaction, llmRequestEvent?.agentProcess?.blackboard)

        // Resolve tool groups and decorate tools
        val tools = resolveAndDecorateTools(interaction, agentProcess, action)

        // Step 1: Original raw chunk stream from LLM
        val rawChunkFlux: Flux<String> = createStreamInternal(
            chatClient = chatClient,
            messages = messages,
            promptContributions = fullPromptContributions,
            tools = tools,
            chatOptions = chatOptions,
            springAiPrompt = springAiPrompt,
        ).filter { it.isNotEmpty() }
            .doOnNext { chunk -> logger.trace("RAW CHUNK: '${chunk.replace("\n", "\\n")}'") }

        // Step 2: Transform raw chunks to complete newline-delimited lines
        val lineFlux: Flux<String> = rawChunkFlux
            .transform { chunkFlux -> rawChunksToLines(chunkFlux) }
            .doOnNext { line -> logger.trace("COMPLETE LINE: '$line'") }

        // Step 3: Final flux of StreamingEvent (thinking + objects)
        val event = lineFlux
            .concatMap { line -> streamingConverter.convertStreamWithThinking(line) }

        return event
    }

    /**
     * Convert raw streaming chunks → NDJSON lines
     * Handles all general cases:
     * - multiple \n in one chunk
     * - no \n in chunk
     * - line spanning many chunks
     */
    fun rawChunksToLines(raw: Flux<String>): Flux<String> {
        val buffer = StringBuilder()
        return raw.concatMap { chunk ->  // ONLY CHANGE: handle → concatMap
            buffer.append(chunk)
            val lines = mutableListOf<String>()
            while (true) {
                val idx = buffer.indexOf('\n')
                if (idx < 0) break
                val line = buffer.substring(0, idx).trim()
                if (line.isNotEmpty()) lines.add(line)
                buffer.delete(0, idx + 1)
            }

            Flux.fromIterable(lines)  // emit multiple lines

        }.doOnComplete {
            // Log any remaining buffer content when stream ends
            if (buffer.isNotEmpty()) {
                val finalLine = buffer.toString().trim()
                if (finalLine.isNotEmpty()) {
                    logger.trace("FINAL LINE: '$finalLine'")
                }
            }
        }.concatWith(
            // final emit
            Mono.fromSupplier { buffer.toString().trim() }
                .filter { it.isNotEmpty() }
        )
    }

    /* -------------------------------------------------------------------------
     * Streaming Abstraction Layer
     *
     * Supports decoupling streaming from Spring AI via LlmMessageStreamer interface.
     * Controlled by useMessageStreamer flag:
     *   - false (default): uses Spring AI ChatClient directly
     *   - true: delegates to vendor-neutral LlmMessageStreamer
     *
     * Enables future support for non-Spring AI providers (e.g., LangChain4j).
     * ------------------------------------------------------------------------ */

    /**
     * Build message list with prompt contributions prepended as system message.
     *
     * Mirrors [buildBasicPrompt] but returns Embabel messages instead of Spring AI Prompt.
     * Used by the decoupled streaming path (when useMessageStreamer=true).
     *
     * @param messages Conversation messages
     * @param promptContributions Prompt contributions to prepend
     * @return Message list with contributions as first system message (if non-empty)
     */
    private fun buildMessagesWithContributions(
        messages: List<Message>,
        promptContributions: String,
    ): List<Message> = buildList {
        if (promptContributions.isNotEmpty()) {
            add(com.embabel.chat.SystemMessage(promptContributions))
        }
        addAll(messages)
    }

    /**
     * Create raw content stream from LLM.
     *
     * Switches between decoupled path (LlmMessageStreamer) and current path (Spring AI direct)
     * based on [useMessageStreamer] flag.
     *
     * @param chatClient Spring AI ChatClient instance
     * @param messages Embabel conversation messages
     * @param promptContributions Prompt contributions string
     * @param tools Embabel tools available for LLM
     * @param chatOptions Spring AI chat options
     * @param springAiPrompt Pre-built Spring AI prompt (used when useMessageStreamer=false)
     * @return Flux of raw content chunks
     */
    private fun createStreamInternal(
        chatClient: org.springframework.ai.chat.client.ChatClient,
        messages: List<Message>,
        promptContributions: String,
        tools: List<com.embabel.agent.api.tool.Tool>,
        chatOptions: org.springframework.ai.chat.prompt.ChatOptions,
        springAiPrompt: Prompt,
    ): Flux<String> {
        return if (useMessageStreamer) {
            val streamerMessages = buildMessagesWithContributions(messages, promptContributions)
            SpringAiLlmMessageStreamer(chatClient, chatOptions).stream(streamerMessages, tools)
        } else {
            chatClient
                .prompt(springAiPrompt)
                .toolCallbacks(tools.toSpringToolCallbacks())
                .options(chatOptions)
                .stream()
                .content()
        }
    }
}

/**
 * Adapter to wrap Spring AI's StructuredOutputConverter as our OutputConverter.
 */
private class SpringAiOutputConverterAdapter<T>(
    private val delegate: org.springframework.ai.converter.StructuredOutputConverter<T>,
) : OutputConverter<T> {
    override fun convert(source: String): T? = delegate.convert(source)
    override fun getFormat(): String? = delegate.format
}
