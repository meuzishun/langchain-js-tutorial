import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';

import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

import * as dotenv from 'dotenv';
dotenv.config();

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
  Answer the user's question.
  Context: {context} 
  Question: {input}
`);

// const chain = prompt.pipe(model);
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

// ## LOAD DATA FROM WEBPAGE
const loader1 = new CheerioWebBaseLoader('https://reactrouter.com/en/main');
const docs1 = await loader1.load();

const loader2 = new CheerioWebBaseLoader(
  'https://reactcommunity.org/react-transition-group/'
);
const docs2 = await loader2.load();

const docs = [...docs1, ...docs2];

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 800,
  chunkOverlap: 20,
});

const splitDocs = await splitter.splitDocuments(docs);
// console.log(splitDocs);

const embeddings = new OpenAIEmbeddings();

const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

// ## RETRIEVE DATA
const retriever = vectorStore.asRetriever({
  k: 2,
});

const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
});

const response = await retrievalChain.invoke({
  input:
    'How can I use the React Transition Group to animate page changes when my routes are stored in an array and passed into createBrowserRouter? Please provide an example. Remember, I am using the createBrowserRouter function and am not using Router, Route, or Switch components.',
});
console.log(response);
