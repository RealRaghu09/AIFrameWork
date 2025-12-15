
import asyncio
import json
import logging
import random
import re
import time
import warnings


from typing import List, Tuple

import aiohttp
import requests
from bs4 import BeautifulSoup



class BaseSearch:

    def __init__(self, topk: int = 3, black_list: List[str] = None):
        self.topk = topk
        self.black_list = black_list

    def _filter_results(self, results: List[tuple]) -> dict:
        filtered_results = {}
        count = 0
        for url, snippet, title in results:
            if all(domain not in url for domain in self.black_list) and not url.endswith('.pdf'):
                filtered_results[count] = {
                    'url': url,
                    'summ': json.dumps(snippet, ensure_ascii=False)[1:-1],
                    'title': title,
                }
                count += 1
                if count >= self.topk:
                    break
        return filtered_results


class DuckDuckGoSearch(BaseSearch):

    def __init__(
        self,
        topk: int = 3,
        black_list: List[str] = [
            'enoN',
            'youtube.com',
            'bilibili.com',
            'researchgate.net',
        ],
        **kwargs,
    ):
        self.proxy = kwargs.get('proxy')
        self.timeout = kwargs.get('timeout', 30)
        super().__init__(topk, black_list)

    def search(self, query: str, max_retry: int = 3) -> dict:
        for attempt in range(max_retry):
            try:
                response = self._call_ddgs(query, timeout=self.timeout, proxy=self.proxy)
                return self._parse_response(response)
            except Exception as e:
                logging.exception(str(e))
                warnings.warn(f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                time.sleep(random.randint(2, 5))
        raise Exception('Failed to get search results from DuckDuckGo after retries.')

    async def asearch(self, query: str, max_retry: int = 3) -> dict:
        from duckduckgo_search import AsyncDDGS

        for attempt in range(max_retry):
            try:
                ddgs = AsyncDDGS(timeout=self.timeout, proxy=self.proxy)
                response = await ddgs.text(query.strip("'"), max_results=10)
                return self._parse_response(response)
            except Exception as e:
                if isinstance(e, asyncio.TimeoutError):
                    logging.exception('Request to DDGS timed out.')
                logging.exception(str(e))
                warnings.warn(f'Retry {attempt + 1}/{max_retry} due to error: {e}')
                await asyncio.sleep(random.randint(2, 5))
        raise Exception('Failed to get search results from DuckDuckGo after retries.')

    async def _async_call_ddgs(self, query: str, **kwargs) -> dict:
        from duckduckgo_search import DDGS

        ddgs = DDGS(**kwargs)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(ddgs.text, query.strip("'"), max_results=10), timeout=self.timeout
            )
            return response
        except asyncio.TimeoutError:
            logging.exception('Request to DDGS timed out.')
            raise

    def _call_ddgs(self, query: str, **kwargs) -> dict:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self._async_call_ddgs(query, **kwargs))
            return response
        finally:
            loop.close()

    def _parse_response(self, response: List[dict]) -> dict:
        raw_results = []
        for item in response:
            raw_results.append(
                (item['href'], item['description'] if 'description' in item else item['body'], item['title'])
            )
        return self._filter_results(raw_results)





class ContentFetcher:

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def fetch(self, url: str) -> Tuple[bool, str]:
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.content
        except requests.RequestException as e:
            return False, str(e)

        text = BeautifulSoup(html, 'html.parser').get_text()
        cleaned_text = re.sub(r'\n+', '\n', text)
        return True, cleaned_text

    async def afetch(self, url: str) -> Tuple[bool, str]:
        try:
            async with aiohttp.ClientSession(
                raise_for_status=True, timeout=aiohttp.ClientTimeout(self.timeout)
            ) as session:
                async with session.get(url) as resp:
                    html = await resp.text(errors='ignore')
                    text = BeautifulSoup(html, 'html.parser').get_text()
                    cleaned_text = re.sub(r'\n+', '\n', text)
                    return True, cleaned_text
        except Exception as e:
            return False, str(e)


