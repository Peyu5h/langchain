
import { config } from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatMistralAI } from "@langchain/mistralai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";
import { SystemMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnableMap } from "@langchain/core/runnables";
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { HuggingFaceInference } from "@langchain/community/llms/hf" ;

config();

const model = new ChatGoogleGenerativeAI({
    modelName: "gemini-1.5-flash-latest"
});

// ====================== Chat Models ======================

async function chat() {
    const chatHistory = [];
    const systemMessage = new SystemMessage("You are a helpful AI assistant.");
    chatHistory.push(systemMessage);

    const rl = readline.createInterface({ input, output });

    try {
        while (true) {
            const query = await rl.question('You: ');
            
            if (query.toLowerCase() === 'exit') break;

            chatHistory.push(new HumanMessage(query));
            const result = await model.invoke(chatHistory);
            chatHistory.push(new AIMessage(result.content));

            console.log(`AI: ${result.content}`);
        }

        console.log('---- Message History ----');
        console.log(chatHistory);
    } finally {
        rl.close();
    }
}

// ====================== Prompt Template ======================

const template = "The user wants to book a flight from {city1} to {city2} on {date}. The user want flight schedule, the price, and the duration of the flight. The assistant should provide the user with the necessary information dummy data for now.";

const promptTemplate = ChatPromptTemplate.fromTemplate(template);

async function testPromptTemplate() {
    const prompt = await promptTemplate.invoke({
        city1: "Mumbai",
        city2: "Delhi",
        date: "2025-03-01"
    });

    const result = await model.invoke(prompt);
    console.log(result.content);
}

// ====================== Chain ================================

async function testChain() {
    const chain = promptTemplate.pipe(model).pipe(new StringOutputParser());
    
    const result = await chain.invoke({
        city1: "mumbai",
        city2: "delhi",
        date: "2025-03-01"
    });
    console.log(result);
}

// ================= Sequential Chaining =====================

const passwordCheckPrompt = ChatPromptTemplate.fromTemplate(`
Check if this password meets basic requirements:
Password: {password}
Requirements:
- At least 8 characters
- Contains numbers
- Contains special characters
Respond with only "PASS" or "FAIL"
`);

const feedbackPrompt = ChatPromptTemplate.fromTemplate(`
Analyze why this password failed and provide feedback:
Password: {password}

Provide two short lines:
1. Why the password failed
2. How to improve it
Keep it brief and clear.
`);

async function createPasswordChain() {
    const passwordValidation = passwordCheckPrompt
        .pipe(model)
        .pipe(new StringOutputParser());

    const feedbackGeneration = feedbackPrompt
        .pipe(model)
        .pipe(new StringOutputParser());

    return async (input) => {
        const checkResult = await passwordValidation.invoke({
            password: input.password
        });

        if (checkResult.trim() === "FAIL") {
            const feedback = await feedbackGeneration.invoke({
                password: input.password
            });
            return `Result: ${checkResult}\nFeedback: ${feedback}`;
        }

        return `Result: ${checkResult}`;
    };
}

async function testPasswordChain() {
    const passwordChain = await createPasswordChain();
    console.log(await passwordChain({ password: "pass123" }));
    console.log(await passwordChain({ password: "P@ssw0rd123!" }));
}

// ================= Parallel Chaining =====================

const sentimentPrompt = ChatPromptTemplate.fromTemplate(`
Analyze the sentiment of this text. 
Text: {text}
Respond with only: POSITIVE, NEGATIVE, or NEUTRAL
`);

const stylePrompt = ChatPromptTemplate.fromTemplate(`
Analyze the writing style of this text.
Text: {text}
Respond with only: FORMAL, CASUAL, or TECHNICAL
`);

function createAnalysisChain() {
    const sentimentAnalysis = sentimentPrompt
        .pipe(model)
        .pipe(new StringOutputParser());

    const styleAnalysis = stylePrompt
        .pipe(model)
        .pipe(new StringOutputParser());

    return RunnableMap.from({
        sentiment: sentimentAnalysis,
        style: styleAnalysis,
        originalText: (input) => input
    });
}

async function testAnalysisChain() {
    const analysisChain = createAnalysisChain();
    const result = await analysisChain.invoke({
        text: "The quarterly financial report indicates a 15% revenue increase."
    });

    console.log(`\nText: ${result.originalText.text}`);
    console.log(`Sentiment: ${result.sentiment}`);
    console.log(`Style: ${result.style}`);
}

// ================= Conditional Chaining =====================

const agePrompt = ChatPromptTemplate.fromTemplate(`
Analyze the age {age} and categorize as:
CHILD (0-12)
TEEN (13-19)
ADULT (20-59)
SENIOR (60+)
Respond with only the category name.
`);

const childPrompt = ChatPromptTemplate.fromTemplate(`
Create a fun, simple message for a child aged {age}.
Include a positive encouragement about learning.
Keep it short and friendly.
`);

const teenPrompt = ChatPromptTemplate.fromTemplate(`
Create a motivational message for a teenager aged {age}.
Focus on growth and future opportunities.
Keep it relatable and brief.
`);

const adultPrompt = ChatPromptTemplate.fromTemplate(`
Provide a professional message for an adult aged {age}.
Focus on work-life balance and personal development.
Keep it concise and meaningful.
`);

const seniorPrompt = ChatPromptTemplate.fromTemplate(`
Create a respectful message for a senior aged {age}.
Focus on wellness and life experience.
Keep it warm and considerate.
`);

function createAgeBasedChain() {
    const ageAnalysis = agePrompt.pipe(model).pipe(new StringOutputParser());
    const childMessage = childPrompt.pipe(model).pipe(new StringOutputParser());
    const teenMessage = teenPrompt.pipe(model).pipe(new StringOutputParser());
    const adultMessage = adultPrompt.pipe(model).pipe(new StringOutputParser());
    const seniorMessage = seniorPrompt.pipe(model).pipe(new StringOutputParser());

    return async (input) => {
        const ageCategory = await ageAnalysis.invoke({ age: input.age });
        let message, category;

        switch (ageCategory.trim()) {
            case "CHILD":
                message = await childMessage.invoke(input);
                category = "Child Message";
                break;
            case "TEEN":
                message = await teenMessage.invoke(input);
                category = "Teen Message";
                break;
            case "ADULT":
                message = await adultMessage.invoke(input);
                category = "Adult Message";
                break;
            default: // SENIOR
                message = await seniorMessage.invoke(input);
                category = "Senior Message";
        }

        return {
            age: input.age,
            category,
            message
        };
    };
}

async function testAgeChain() {
    const ageChain = createAgeBasedChain();
    const result = await ageChain({ age: 70 });
    console.log(`Age: ${result.age}`);
    console.log(`Category: ${result.category}`);
    console.log(`Message: ${result.message}`);
}

async function main() {
    await chat();
    // await testPromptTemplate();
    // await testChain();
    // await testPasswordChain();
    // await testAnalysisChain();
    // await testAgeChain();
}

main().catch(console.error);