from copy import copy
from typing import Dict, List, Optional, Tuple, Union


class LMTemplateParser:
    """Intermidate prompt template parser, specifically for language models."""

    def __init__(self, meta_template: Optional[List[Dict]] = None):
        self.meta_template = meta_template
        if meta_template:
            assert isinstance(meta_template, list)
            self.roles: Dict[str, dict] = dict()  # maps role name to config
            for item in meta_template:
                assert isinstance(item, dict)
                assert item['role'] not in self.roles, \
                    'role in meta prompt must be unique!'
                self.roles[item['role']] = item.copy()

    def __call__(self, dialog) -> str:
        """Parse a prompt template, and wrap it with meta template if
        applicable.
        """
        assert isinstance(dialog, (str, list))
        if isinstance(dialog, str):
            return dialog
        if self.meta_template:

            prompt = ''
            for index, item in enumerate(dialog):
                if isinstance(item, str):
                    prompt += item
                else:
                    new_str = self._prompt2str(item, index == len(dialog) - 1)
                    prompt += new_str
        return prompt

    def _format_begin(self, role_cfg, message):
        name = message.get('name', None)
        if name is not None:
            begin = role_cfg['begin'].get('with_name', '')
            if name in role_cfg['begin'].get('name', {}):
                begin = begin.format(name=role_cfg['begin']['name'][name])
            else:
                begin = begin.format(name=name)
        else:
            if isinstance(role_cfg.get('begin', ''), str):
                begin = role_cfg.get('begin', '')
            elif isinstance(role_cfg['begin'], dict):
                begin = role_cfg['begin'].get('without_name', '')
        return begin

    def _prompt2str(self,
                    prompt: Union[str, Dict],
                    last: bool = False) -> Tuple[str, bool]:
        if isinstance(prompt, str):
            return prompt
        merged_prompt = self.roles.get(prompt['role'])

        if merged_prompt.get('fallback_role'):
            merged_prompt = self.roles.get(merged_prompt['fallback_role'])
        begin = self._format_begin(merged_prompt, prompt)
        res = begin
        if last and merged_prompt.get('generate', False):
            res += prompt.get('content', '')
            return res
        res += prompt.get('content', '') + merged_prompt.get('end', '')
        if last and merged_prompt['role'] != 'assistant':
            res += self._format_begin(self.roles['assistant'], {})
            return res
        return res


class BaseLLM:
    """Base class for model wrapper.
    """

    def __init__(self,
                 path: str,
                 tokenizer_only: bool = False,
                 template_parser: 'LMTemplateParser' = LMTemplateParser,
                 meta_template: Optional[List[Dict]] = None,
                 *,
                 max_new_tokens: int = 512,
                 top_p: float = 0.8,
                 top_k: float = 40,
                 temperature: float = 0.8,
                 repetition_penalty: float = 1.0,
                 stop_words: Union[List[str], str] = None):
        self.path = path
        self.tokenizer_only = tokenizer_only
        # meta template
        self.template_parser = template_parser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

        if isinstance(stop_words, str):
            stop_words = [stop_words]
        self.gen_params = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stop_words=stop_words)

   



    def chat(self,
             inputs: Union[List[dict], List[List[dict]]],
             **gen_params):
        """Generate completion from a list of templates
        """
        if isinstance(inputs[0], list):
            _inputs = list()
            for msg in inputs:
                _inputs.append(self.template_parser(msg))
        else:
            _inputs = self.template_parser(inputs)
        return self.generate(_inputs, **gen_params)
