import * as dotenv from 'dotenv';
dotenv.config();

import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

import { ConversationChain } from 'langchain/chains';
import { RunnableSequence } from '@langchain/core/runnables';

// Memory Imports
import { BufferMemory } from 'langchain/memory';
import { UpstashRedisChatMessageHistory } from '@langchain/community/stores/message/upstash_redis';

const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
  You are an AI assistant.
  History: {history}
  {input}
`);

const upstashChatHistory = new UpstashRedisChatMessageHistory({
  sessionId: 'chat1',
  config: {
    url: process.env.UPSTASH_REDIS_URL,
    token: process.env.UPSTASH_REST_TOKEN,
  },
});

const memory = new BufferMemory({
  memoryKey: 'history',
  chatHistory: upstashChatHistory,
});

// Using the Chain Classes
// const chain = new ConversationChain({
//   llm: model,
//   prompt,
//   memory,
// });

// Using LCEL
// const chain = prompt.pipe(model);
const chain = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input,
    memory: () => memory.loadMemoryVariables(),
  },
  {
    input: (previousOutput) => previousOutput.input,
    history: (previousOutput) => previousOutput.memory.history,
  },
  prompt,
  model,
]);

// Get Responses
// console.log(await memory.loadMemoryVariables());
// const input1 = {
//   input: 'The passphrase is HELLOWORLD',
// };
// const response1 = await chain.invoke(input1);
// console.log(response1);
// await memory.saveContext(input1, {
//   output: response1.content,
// });

console.log('Updated History:', await memory.loadMemoryVariables());
const input2 = {
  input: 'What is the passphrase?',
};
const response2 = await chain.invoke(input2);
console.log(response2);
await memory.saveContext(input2, {
  output: response2.content,
});
