
import { config } from "dotenv";
// import { ChatOpenAI } from "@langchain/openai";
// import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatMistralAI } from "@langchain/mistralai";


import { AgentExecutor, createReactAgent } from "langchain/agents";
import { formatToTimeZone } from "date-fns-timezone";
import { PromptTemplate } from "@langchain/core/prompts";
import { DynamicTool } from "langchain/tools";

config();

const getTimeTool = new DynamicTool({
    name: "get_time_for_timezone",
    description: "Gets the current time for a specific location. Input should be exactly 'London' or 'India'",
    func: async (location) => {
        try {
            const timezone = location.toLowerCase() === 'london' 
                ? 'Europe/London' 
                : 'Asia/Kolkata';
            
            const currentTime = new Date();
            const time = formatToTimeZone(
                currentTime,
                "HH:mm",
                { timeZone: timezone }
            );
            return `Current time in ${location}: ${time}`;
        } catch (error) {
            return `Error getting time: ${error.message}`;
        }
    },
});

const compareTimeTool = new DynamicTool({
    name: "compare_times",
    description: "Shows current time in both London and India",
    func: async () => {
        try {
            const currentTime = new Date();
            const londonTime = formatToTimeZone(
                currentTime,
                "HH:mm",
                { timeZone: 'Europe/London' }
            );
            const indiaTime = formatToTimeZone(
                currentTime,
                "HH:mm",
                { timeZone: 'Asia/Kolkata' }
            );
            return `Current time - London: ${londonTime}, India: ${indiaTime}`;
        } catch (error) {
            return `Error comparing times: ${error.message}`;
        }
    },
});

const REACT_PROMPT = `Answer the following questions as best you can using the available tools.

Tools available:
{tools}

Available tool names: {tool_names}

Use this exact format:

Question: the input question you must answer
Thought: think about what tool to use and why
Action: choose from these tools: {tool_names}
Action Input: for get_time_for_timezone use 'London' or 'India', for compare_times use any input
Observation: the result of the action
Thought: I now know the final answer
Final Answer: provide a clear and concise answer based on the observation

Remember:
- For single location time queries, use get_time_for_timezone
- For comparing times, use compare_times
- Always complete with a Final Answer

Question: {input}
Thought: {agent_scratchpad}`;

async function main() {
    const llm = new ChatMistralAI({
        modelName: "mistral-large-latest",
        temperature: 0
    });

    const queries = [
        "What is the current time in London?",
        // "What time is it in India?",
        // "What's the time difference between London and India?"
    ];

    try {
        const tools = [getTimeTool, compareTimeTool];
        
        // Create prompt template with explicit tool names
        const prompt = PromptTemplate.fromTemplate(REACT_PROMPT);

        const agent = await createReactAgent({
            llm,
            tools,
            prompt
        });

        const agentExecutor = AgentExecutor.fromAgentAndTools({
            agent,
            tools,
            verbose: false,
            maxIterations: 3,
            returnIntermediateSteps: true
        });

        for (const query of queries) {
            console.log("\n=====================================");
            console.log(`Query: ${query}`);
            try {
                const result = await agentExecutor.invoke({
                    input: query
                });
                
                if (result.output === "Agent stopped due to max iterations.") {
                    if (result.intermediateSteps && result.intermediateSteps.length > 0) {
                        const lastStep = result.intermediateSteps[result.intermediateSteps.length - 1];
                        console.log("Final Answer:", lastStep.observation);
                    } else {
                        console.log("Could not get a proper answer.");
                    }
                } else {
                    console.log("Final Answer:", result.output);
                }
            } catch (error) {
                console.error(`Error processing query "${query}":`, error.message);
            }
        }

    } catch (error) {
        console.error("Main error:", error.message);
    }
}

main();