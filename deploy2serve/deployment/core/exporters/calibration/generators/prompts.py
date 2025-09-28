import aiohttp
import asyncio
from deploy2serve.deployment.core.exporters.calibration.generators.interface import LabelsGenerator
import subprocess
import re
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
import yaml


class PromptGenerator(LabelsGenerator):
    def __init__(self, dataset_folder: Union[str, Path]) -> None:
        super().__init__(dataset_folder)
        self.target_count = 1000 # final number of prompts
        self.batch_size = 50  # Prompts per request
        self.concurrent_requests = 3
        self.model: str = "mistral"
        self.timeout = aiohttp.ClientTimeout(total=120)
        self.generation_prompt: str = f"""Generate exactly {self.batch_size} unique image generation prompts. Each 
        prompt must be on a separate line. Format: [Genre/Theme] [Main Subject], [Art Style], [Technical Details]."""


    @staticmethod
    def _filter_prompts(content: str) -> List[str]:
        prompts: List[str] = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()

            if not line or len(line) < 20:
                continue

            # Remove the numbering (1., 2., 3., etc.)
            if re.match(r'^\d+[\.\)]\s*', line):
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
            # Remove list markers (-, *, •)
            line = re.sub(r'^[\-\*•]\s*', '', line)
            # Remove quotation marks at the beginning/end
            line = re.sub(r'^["\']|["\']$', '', line).strip()
            prompts.append(line)

        return prompts

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _generate_with_fallback(self, session: aiohttp.ClientSession, batch_id: int) -> List[str]:
        try:
            async with session.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": self.generation_prompt}],
                        "options": {"temperature": 0.9, "num_predict": 2000},
                        "stream": False
                    },
                    timeout=self.timeout
            ) as response:
                if response.status != 200:
                    raise aiohttp.ClientError(f"HTTP {response.status}")

                data = await response.json()
                content = data.get("message", {}).get("content", "")

                if not content:
                    raise ValueError("Empty response content")

                prompts = self._filter_prompts(content)
                return prompts

        except Exception as error:
            self.logger.warning(f"Batch {batch_id} failed: {error}")
            raise

    async def _generate_with_retry(self, session: aiohttp.ClientSession, batch_id: int) -> List[str]:
        for attempt in range(3):
            try:
                prompts = await self._generate_with_fallback(session, batch_id)
                if len(prompts) >= self.batch_size * 0.8:
                    return prompts[:self.batch_size]

                self.logger.warning(f"Batch {batch_id} attempt {attempt + 1}: generated only {len(prompts)} prompts")
            except Exception as error:
                self.logger.warning(f"Batch {batch_id} attempt {attempt + 1} failed: {error}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        return []

    async def _async_generate(self) -> List[str]:
        prompts: List[str] = []
        batches = (self.target_count + self.batch_size - 1) // self.batch_size

        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.concurrent_requests)

            async def limited_generation(batch_id):
                async with semaphore:
                    return await self._generate_with_retry(session, batch_id)

            tasks = [limited_generation(i) for i in range(batches)]

            completed_batches = 0
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating batches"):
                try:
                    batch_prompts = await future
                    if batch_prompts:
                        prompts.extend(batch_prompts)
                        completed_batches += 1

                    if completed_batches % 2 == 0 and prompts:
                        await self._async_dump(prompts)

                    if len(prompts) >= self.target_count:
                        break

                except Exception as error:
                    self.logger.error(f"Batch processing failed: {error}")
                    continue

            [task.close() for task in tasks]
        return prompts

    async def _async_dump(self, prompts: List[str]) -> None:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._save_cache(prompts))
        except Exception as error:
            self.logger.error(f"Async save failed: {error}")

    def _save_cache(self, prompts: List[str]) -> None:
        cache = self.dataset_folder.joinpath("prompts.yaml")
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            with cache.open("w", encoding="utf-8") as file:
                yaml.safe_dump(prompts, file, allow_unicode=True)
        except Exception as error:
            self.logger.error(f"Sync save failed: {error}")

    def _load_cache(self) -> Optional[List[str]]:
        cache = self.dataset_folder.joinpath("prompts.yaml")
        if cache.exists():
            try:
                with cache.open("r", encoding="utf-8") as file:
                    prompts = yaml.safe_load(file)
                    if prompts and len(prompts) >= self.target_count:
                        self.logger.info(f"Loaded {len(prompts)} prompts from cache")
                        return prompts[:self.target_count]
            except Exception as error:
                self.logger.warning(f"Cache load failed: {error}")
        return None

    def generate_labels(self) -> Optional[Dict[str, Any]]:
        cached_prompts = self._load_cache()
        if cached_prompts:
            return {"prompts": cached_prompts}

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            prompts = loop.run_until_complete(self._async_generate())
        except Exception as error:
            self.logger.error(f"Async generation failed: {error}")
            prompts = []
        finally:
            loop.close()
            subprocess.run(
                ["ollama", "stop", self.model],
                capture_output=True,
                text=True,
                timeout=10
            )

        if prompts:
            self._save_cache(prompts)
            self.logger.info(f"Successfully generated {len(prompts)} prompts")
        else:
            self.logger.error("No prompts were generated")
            prompts = []

        return {"prompts": prompts[:self.target_count]}
