import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import * as dotenv from 'dotenv';
dotenv.config();

// Create model
const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
});

// Create Prompt Template
const prompt = ChatPromptTemplate.fromMessages([
  ['system', 'Generate a joke based on a word provided by the user.'],
  ['human', '{input}'],
]);

// Create chain
const chain = prompt.pipe(model);

// Call chain
const response = await chain.invoke({ input: 'dog' });

console.log(response);
