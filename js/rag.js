
import { config } from "dotenv";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import fs from "fs/promises";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";

config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const filePath = join(__dirname, "documents", "Dracula.txt");

async function readTextFile(filePath) {
    const encodings = ['utf-8', 'latin1', 'ascii'];
    
    for (const encoding of encodings) {
        try {
            const content = await fs.readFile(filePath, { encoding });
            return content;
        } catch (error) {
            if (error.code === 'ERR_ENCODING_NOT_SUPPORTED') continue;
            throw error;
        }
    }
    
    throw new Error(`Could not read file with any of the encodings: ${encodings}`);
}

async function initializeVectorStore() {
    console.log("Creating new vector store...");
    
    try {
        await fs.access(filePath);        
        const textContent = await readTextFile(filePath);
        const documents = [new Document({ pageContent: textContent })];
        
        const textSplitter = new CharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 100,
            separator: "\n"
        });
        
        const docs = await textSplitter.splitDocuments(documents);
        console.log(`Split document into ${docs.length} chunks`);
        
        const embeddings = new GoogleGenerativeAIEmbeddings({
            modelName: "models/embedding-001"
        });
        
        // in-memory vector store
        const vectorStore = await MemoryVectorStore.fromDocuments(
            docs,
            embeddings
        );
        
        console.log("Vector store created successfully!");
        return vectorStore;
        
    } catch (error) {
        console.error("Error initializing vector store:", error);
        throw error;
    }
}

async function queryDatabase(vectorStore, query) {
    try {
        const relevantDocs = await vectorStore.similaritySearch(query, 5);
        
        console.log("\n--- Relevant Documents ---");
        relevantDocs.forEach((doc, i) => {
            console.log(`\nDocument ${i + 1}:`);
            console.log(doc.pageContent.trim());
        });
        
        const context = relevantDocs
            .map(doc => doc.pageContent.trim())
            .join("\n\n");
            
        const combinedInput = `
        Question: ${query}
        
        Context from documents:
        ${context}
        
        Please answer the question based only on the provided context. 
        If the answer isn't clear from the context, say "I'm not sure."
        Provide a concise answer with relevant quotes if available.
        `;
        
        const model = new ChatGoogleGenerativeAI({
            modelName: "gemini-1.5-flash-latest"
        });
        
        const messages = [
            new SystemMessage("You are a helpful assistant specialized in analyzing literature."),
            new HumanMessage(combinedInput)
        ];
        
        const result = await model.invoke(messages);
        return result.content;
        
    } catch (error) {
        return `Error querying database: ${error.message}`;
    }
}

async function main() {
    try {
        const vectorStore = await initializeVectorStore();
        
        const readline = (await import('readline')).createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        const askQuestion = (query) => new Promise((resolve) => {
            readline.question(query, resolve);
        });
        
        while (true) {
            const query = await askQuestion("\nEnter your question (or 'quit' to exit): ");
            
            if (query.toLowerCase() === 'quit') break;
            
            console.log(`\nQuery: ${query}`);
            const response = await queryDatabase(vectorStore, query);
            console.log("\n--- Generated Response ---");
            console.log(response);
        }
        
        readline.close();
        console.log("\nThank you for using the Dracula QA system!");
        
    } catch (error) {
        console.error("An error occurred:", error);
    }
}

main();