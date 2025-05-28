from typing import List, TypedDict, Any, Dict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime, timezone
import torch
import traceback
import re

import asyncio
generator_lock = asyncio.Lock()



# Current UTC time for the tool
def get_current_time() -> str:
    return datetime.now(timezone.utc).isoformat()

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# System prompt with clear instructions
system_prompt = SystemMessage(content="""
Вы — точный и полезный ассистент по имени Qwen. Когда вас спрашивают текущее время, не называйте его, а используйте следующий вызов инструмента вместо названия точного времени:
[TOOL_CALL]get_current_time[/TOOL_CALL].
И больше ничего не добавляйте. Во всех остальных случаях отвечайте полно и по существу.
Пример:
                              [User]: Сколько сейчас времени?
                              [Ассистент]: Точное время сейчас: [TOOL_CALL]get_current_time[/TOOL_CALL]
""".strip())

# Format prompt: system message + last user message
def format_prompt(last_user: HumanMessage) -> str:
    prompt = f"[User] {last_user}\n"
    return prompt

class State(TypedDict):
    messages: List[BaseMessage]

def generate_response(state: State, config: RunnableConfig) -> Dict[str, Any]:
    if 'messages' not in state:
        state['messages'] = [system_prompt]
    last_user = None
    for msg in reversed(state['messages']):
        import json
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get('type') == 'human'):
            last_user = msg['content']
            break
    if last_user is None:
        ai_msg = AIMessage(json.dumps(msg).split('[User]')[0])
        state['messages'].append(ai_msg)
        return {'messages': state['messages']}

    prompt = format_prompt(last_user)
    try:
        with torch.no_grad():
            messages = [
            {"role": "system", "content": system_prompt.content},
            {"role": "user", "content": prompt}
        ]
            text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated = out
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = (
            "Ошибка в text_generator:\n",
            f"{str(e)}\n\n{tb}"
        )
        state['messages'].append(AIMessage(content=err_msg))
        return {'messages': state['messages']}

    ai_msg = AIMessage(content=generated)
    state['messages'].append(ai_msg)
    return {'messages': state['messages']}

def tool_execution(state: State, config: RunnableConfig) -> Dict[str, Any]:
    last_ai = state['messages'][-1]
    if isinstance(last_ai, AIMessage) and isinstance(last_ai.content, str) and "[TOOL_CALL]" in last_ai.content:
        original = last_ai.content
        def replacer(match):
            tool_name = match.group(1)
            if tool_name == 'get_current_time':
                return get_current_time()
            return match.group(0)
        new_content = re.sub(r"\[TOOL_CALL](.+?)\[/TOOL_CALL]", replacer, original)
        if new_content != original:
            last_ai.content = new_content
    return {'messages': state['messages']}

def route(state: State) -> str:
    last = state['messages'][-1]
    if isinstance(last, AIMessage):
        if "[TOOL_CALL]" in last.content:
            return 'tool'
        else:
            return END  
    return END

graph = StateGraph(State)
graph.add_node('generate', generate_response)
graph.add_node('tool', tool_execution)
graph.add_edge('__start__', 'generate')
graph.add_conditional_edges('generate', route, {'tool': 'tool', END: END})
graph.add_conditional_edges('tool', route, {'tool': 'tool', END: END})