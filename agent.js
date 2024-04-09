import * as dotenv from 'dotenv';
dotenv.config();

import readline from 'readline';

import { ChatOpenAI } from '@langchain/openai';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import { createOpenAIFunctionsAgent, AgentExecutor } from 'langchain/agents';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { createRetrieverTool } from 'langchain/tools/retriever';

import { HumanMessage, AIMessage } from '@langchain/core/messages';

import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

const loader = new CheerioWebBaseLoader(
  'https://js.langchain.com/docs/expression_language/'
);
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});

const splitDocs = await splitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings();

const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

const retriever = vectorStore.asRetriever({
  k: 2,
});

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo-1106',
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromMessages([
  ('system', 'You are a helpful assistant called Max.'),
  new MessagesPlaceholder('chat_history'),
  ('human', '{input}'),
  new MessagesPlaceholder('agent_scratchpad'),
]);

// Create and Assign Tools
const searchTool = new TavilySearchResults();
const retrieverTool = createRetrieverTool(retriever, {
  name: 'lcel_search',
  description:
    'Use this tools when searching for information about LangChain Expression Language (LCEL)',
});
const tools = [searchTool, retrieverTool];

// Create Agent
const agent = await createOpenAIFunctionsAgent({
  llm: model,
  prompt,
  tools,
});

// Create Agent Executor
const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

// Get user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chatHistory = [];

const askQuestion = () => {
  rl.question('User: ', async (input) => {
    if (input.toLowerCase() === 'exit') {
      rl.close();
      return;
    }
    // Call Agent
    const response = await agentExecutor.invoke({
      input: input,
      chat_history: chatHistory,
    });

    console.log('Agent: ', response.output);
    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(response.output));

    askQuestion();
  });
};

askQuestion();
