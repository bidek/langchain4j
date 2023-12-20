package dev.langchain4j.model.ollama;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import lombok.Builder;
import org.jetbrains.annotations.NotNull;

import java.time.Duration;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static dev.langchain4j.internal.RetryUtils.withRetry;
import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.ValidationUtils.ensureNotBlank;

/**
 * Ollama chat model implementation.
 */
public class OllamaChatModel implements ChatLanguageModel {

    private final OllamaClient client;
    private final Double temperature;
    private final String modelName;
    private final Integer maxRetries;

    @Builder
    public OllamaChatModel(String baseUrl, Duration timeout,
                           String modelName, Double temperature, Integer maxRetries) {
        this.client = OllamaClient.builder().baseUrl(baseUrl).timeout(timeout).build();
        this.modelName = ensureNotBlank(modelName, "modelName");
        this.temperature = getOrDefault(temperature, 0.7);
        this.maxRetries = getOrDefault(maxRetries, 3);
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages) {
        throwExceptionIfNullOrEmpty(messages);

        List<Message> messageList = prepareMessageToSendWithRequest(messages);

        ChatRequest request = prepareOllamaChatRequest(messageList);
        ChatResponse response = withRetry(() -> client.chat(request), maxRetries);

        return Response.from(AiMessage.from(response.getMessage().getContent()));
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages, List<ToolSpecification> toolSpecifications) {
        throwExceptionIfNullOrEmpty(messages);

        List<Message> messageList = prepareMessageToSendWithRequest(messages);
        List<Message> howToUseToolsSystemMessageList = Collections.singletonList(prepareSystemMessageWithToolUsage(toolSpecifications));
        List<Message> systemMessageMessagesWithToolDescriptions = prepereToolDesciptionMessageList(toolSpecifications);

        List<Message> systemToolsMessages = Stream.concat(
                        howToUseToolsSystemMessageList.stream(),
                        systemMessageMessagesWithToolDescriptions.stream())
                .collect(Collectors.toList());

        List<Message> combinedMessages = Stream.concat(
                        systemToolsMessages.stream(),
                        messageList.stream())
                .collect(Collectors.toList());


        ChatRequest request = prepareOllamaChatRequest(combinedMessages);
        ChatResponse response = withRetry(() -> client.chat(request), maxRetries);

        String aiTextResponse = response.getMessage().getContent();
        if (aiTextResponse.contains("TOOL-EXECUTION-REQUEST")) {
            return Response.from(
                    AiMessage.aiMessage(prepareToolExecutionRequest(aiTextResponse)),
                    new TokenUsage(0, 0)
            );
        } else {
            return Response.from(AiMessage.from(aiTextResponse),
                    new TokenUsage(0, 0)
            );
        }
    }

    private ToolExecutionRequest prepareToolExecutionRequest(String aiTextResponse) {
        String regexp = "TOOL-EXECUTION-REQUEST\\s-\\s(\\w*)";
        Matcher matcher = Pattern.compile(regexp).matcher(aiTextResponse);
        if (matcher.find()) {
            return ToolExecutionRequest.builder()
                    .name(matcher.group(1))
                    .build();

        } else {
            throw new RuntimeException("Ai respond with tool execution request, cant decode requested tool name, AI response: " + aiTextResponse);
        }
    }

    @NotNull
    private List<Message> prepereToolDesciptionMessageList(List<ToolSpecification> toolSpecifications) {
        return toolSpecifications.stream().map(toolSpecification ->
                Message.builder()
                        .role(Role.ASSISTANT)
                        .content(renderToolDescription(toolSpecification))
                        .build()
        ).collect(Collectors.toList());
    }

    private ChatRequest prepareOllamaChatRequest(List<Message> messageList) {
        return ChatRequest.builder()
                .model(modelName)
                .messages(messageList)
                .options(Options.builder()
                        .temperature(temperature)
                        .build())
                .stream(false)
                .build();
    }

    @NotNull
    private List<Message> prepareMessageToSendWithRequest(List<ChatMessage> messages) {
        return messages.stream()
                .map(this::createMessageFromChatMessage)
                .collect(Collectors.toList());
    }


    private Message createMessageFromChatMessage(ChatMessage chatMessage) {
        if (chatMessage instanceof AiMessage && ((AiMessage) chatMessage).hasToolExecutionRequests()) {
            AiMessage aiMessage = (AiMessage) chatMessage;
            String requestedToolName = aiMessage.toolExecutionRequests().get(0).name();

            return Message.builder()
                    .role(Role.ASSISTANT)
                    .content("AI requested TOOL-EXECUTION-REQUEST for TOOL: " + requestedToolName)
                    .build();

        } else if (chatMessage instanceof ToolExecutionResultMessage) {
            ToolExecutionResultMessage toolExecutionResultMessage = (ToolExecutionResultMessage) chatMessage;

            return Message.builder()
                    .role(Role.SYSTEM)
                    .content("TOOL-EXECUTION-RESULT for TOOL: " + toolExecutionResultMessage.toolName() + ", RESULT: " + toolExecutionResultMessage.text())
                    .build();
        } else {
            return Message.builder()
                    .role(Role.fromChatMessageType(chatMessage.type()))
                    .content(chatMessage.text())
                    .build();
        }
    }

    private void throwExceptionIfNullOrEmpty(List<ChatMessage> messages) {
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("messages must not be null or empty");
        }
    }

    private Message prepareSystemMessageWithToolUsage(List<ToolSpecification> toolSpecificationList) {
        String toolList = toolSpecificationList.stream().map(ToolSpecification::name).collect(Collectors.joining(", "));
        Map<String, Object> systemMessageConentMap = new HashMap<String, Object>() {{
            put("toolList", toolList);
        }};

        String messageHowToUseTool = "I, as AI, am capable of using some TOOLS, here is a list of all TOOLS I can use: {{toolList}}. If USER ask me for something and in conclusion I must use TOOL, I must respond exactly like this: 'TOOL-EXECUTION-REQUEST - [TOOL_NAME]'. Other SYSTEM will respond for my TOOL-EXECUTION-REQUEST and provide me TOOL-EXECUTION-RESULT with a result of my TOOL-EXECUTION-REQUEST in a next message. I use TOOLS only when i really need. I never tell USER that i made TOOL-EXECUTION-REQUEST";

        return Message.builder()
                .role(Role.ASSISTANT)
                .content(
                        PromptTemplate.from(messageHowToUseTool)
                                .apply(systemMessageConentMap)
                                .text())
                .build();
    }

    private String renderToolDescription(ToolSpecification toolSpecification) {
        Map<String, Object> toolDescriptionMap = new HashMap<String, Object>() {{
            put("toolName", toolSpecification.name());
            put("toolDesc", toolSpecification.description());
            put("toolParams", "");
        }};

        return PromptTemplate.from("TOOL: {{toolName}}, TOOL DESCRIPTION: {{toolDesc}}{{toolParams}}")
                .apply(toolDescriptionMap)
                .text();
    }
}
