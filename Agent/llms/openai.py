import json
import os
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock
from typing import  Dict, List, Optional, Union
import requests

from .base_api import BaseAPILLM


OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'


class GPTAPI(BaseAPILLM):
    """Model wrapper around OpenAI's models."""

    is_api: bool = True

    def __init__(
        self,
        model_type: str = 'gpt-4o-mini',# Model type
        retry: int = 2,# Number of retries if the API call fails
        json_mode: bool = False, # Response format
        key: Union[str, List[str]] = 'ENV',#API key
        meta_template: Optional[Dict] = [ # Meta data 
            dict(role='system', api_role='system'),
            dict(role='user', api_role='user'),
            dict(role='assistant', api_role='assistant'),
            dict(role='environment', api_role='system'),
        ],
        api_base: str = OPENAI_API_BASE, # API base URL
        **gen_params, # Generation parameters
    ):
        super().__init__(model_type=model_type, meta_template=meta_template, retry=retry, **gen_params)
        self.gen_params.pop('top_k')
        self.logger = getLogger(__name__)

        if isinstance(key, str):
            self.keys = [os.getenv('OPENAI_API_KEY') if key == 'ENV' else key]
        else:
            self.keys = key
        self.invalid_keys = set()

        self.key_ctr = 0
        self.url = api_base
        self.model_type = model_type
        self.json_mode = json_mode

    def chat(
        self,
        inputs: Union[List[dict], List[List[dict]]], # Input messages
    ) -> Union[str, List[str]]: # Generated responses
        """Generate responses given the contexts."""
        assert isinstance(inputs, list)
        with ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [
                executor.submit(self._chat, messages)
                for messages in ([inputs] if isinstance(inputs[0], dict) else inputs)
            ]
        ret = [task.result() for task in tasks]
        return ret[0] if isinstance(inputs[0], dict) else ret

    def _chat(self, messages: List[dict], **gen_params) -> str:
        """Generate completion from a list of templates."""
        assert isinstance(messages, list)
        messages = self.template_parser(messages)
        header, data = self.generate_request_data(
            model_type=self.model_type, messages=messages, gen_params=gen_params, json_mode=self.json_mode
        )

        max_num_retries, errmsg = 0, ''
        while max_num_retries < self.retry:
            with Lock():
                if len(self.invalid_keys) == len(self.keys):
                    raise RuntimeError('All keys have insufficient quota.')

                # find the next valid key
                while True:
                    self.key_ctr += 1
                    if self.key_ctr == len(self.keys):
                        self.key_ctr = 0

                    if self.keys[self.key_ctr] not in self.invalid_keys:
                        break

                key = self.keys[self.key_ctr]
                header['Authorization'] = f'Bearer {key}'

            if self.orgs:
                with Lock():
                    self.org_ctr += 1
                    if self.org_ctr == len(self.orgs):
                        self.org_ctr = 0
                header['OpenAI-Organization'] = self.orgs[self.org_ctr]

            response = dict()
            try:
                raw_response = requests.post(self.url, headers=header, data=json.dumps(data))
                response = raw_response.json()
                return response['choices'][0]['message']['content'].strip()
            except requests.ConnectionError:
                errmsg = 'Got connection error ' + str(traceback.format_exc())
                self.logger.error(errmsg)
                continue
            except requests.JSONDecodeError:
                errmsg = 'JsonDecode error, got ' + str(raw_response.content)
                self.logger.error(errmsg)
                continue
            except KeyError:
                if 'error' in response:
                    if response['error']['code'] == 'rate_limit_exceeded':
                        time.sleep(1)
                        continue
                    elif response['error']['code'] == 'insufficient_quota':
                        self.invalid_keys.add(key)
                        self.logger.warn(f'insufficient_quota key: {key}')
                        continue

                    errmsg = 'Find error message in response: ' + str(response['error'])
                    self.logger.error(errmsg)
            except Exception as error:
                errmsg = str(error) + '\n' + str(traceback.format_exc())
                self.logger.error(errmsg)
            max_num_retries += 1

        raise RuntimeError(
            'Calling OpenAI failed after retrying for '
            f'{max_num_retries} times. Check the logs for '
            f'details. errmsg: {errmsg}'
        )

    def generate_request_data(self, model_type, messages, gen_params, json_mode=False):
        """Generates the request data for different model types"""
        # Copy generation parameters to avoid modifying the original dictionary
        gen_params = gen_params.copy()

        # Hold out 100 tokens due to potential errors in token calculation
        max_tokens = min(gen_params.pop('max_new_tokens'), 4096)
        if max_tokens <= 0:
            return '', ''

        header = {
            'content-type': 'application/json',
        }

        gen_params['max_tokens'] = max_tokens
        if 'stop_words' in gen_params:
            gen_params['stop'] = gen_params.pop('stop_words')
        if 'repetition_penalty' in gen_params:
            gen_params['frequency_penalty'] = gen_params.pop('repetition_penalty')
        data = {}
        if model_type.lower().startswith('gpt') or model_type.lower().startswith('qwen'):
            if 'top_k' in gen_params:
                warnings.warn('`top_k` parameter is deprecated in OpenAI APIs.', DeprecationWarning)
                gen_params.pop('top_k')
            gen_params.pop('skip_special_tokens', None)
            gen_params.pop('session_id', None)
            data = {'model': model_type, 'messages': messages, 'n': 1, **gen_params}
            if json_mode:
                data['response_format'] = {'type': 'json_object'}
        elif model_type.lower().startswith('internlm'):
            data = {'model': model_type, 'messages': messages, 'n': 1, **gen_params}
            if json_mode:
                data['response_format'] = {'type': 'json_object'}
        else:
            raise NotImplementedError(f'Model type {model_type} is not supported')

        return header, data

    def tokenize(self, prompt: str) -> list:
        """Tokenize the input prompt.
        """
        import tiktoken

        self.tiktoken = tiktoken
        enc = self.tiktoken.encoding_for_model(self.model_type)
        
        return enc.encode(prompt)


