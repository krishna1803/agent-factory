"""
Enhanced LLM provider implementations for OpenAI, Ollama, and OCI GenAI.

This module provides concrete implementations for connecting to various
LLM providers with proper configuration management and error handling.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from agent_factory.core.models import AgentSpec, ModelProvider, ModelProfile

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Custom exception for LLM provider errors."""
    pass


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model_profile: ModelProfile):
        """Initialize the LLM provider."""
        self.model_profile = model_profile
        self.llm = None
        self.embeddings = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM provider."""
        pass
    
    @abstractmethod
    async def generate_response(self, messages: List[BaseMessage]) -> AIMessage:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the given texts."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, model_profile: ModelProfile):
        """Initialize OpenAI provider."""
        super().__init__(model_profile)
        self.api_key = model_profile.api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = model_profile.api_base or os.getenv("OPENAI_API_BASE")
        
        if not self.api_key:
            raise LLMProviderError("OpenAI API key is required")
    
    async def initialize(self) -> None:
        """Initialize OpenAI models."""
        try:
            # Initialize chat model
            self.llm = ChatOpenAI(
                model=self.model_profile.model_name,
                temperature=self.model_profile.temperature,
                max_tokens=self.model_profile.max_tokens,
                top_p=self.model_profile.top_p,
                frequency_penalty=self.model_profile.frequency_penalty,
                presence_penalty=self.model_profile.presence_penalty,
                api_key=self.api_key,
                base_url=self.api_base,
                **self.model_profile.additional_config
            )
            
            # Initialize embeddings model
            embedding_model = self.model_profile.additional_config.get(
                'embedding_model', 'text-embedding-ada-002'
            )
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                api_key=self.api_key,
                base_url=self.api_base
            )
            
            logger.info(f"Initialized OpenAI provider with model: {self.model_profile.model_name}")
            
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize OpenAI provider: {str(e)}")
    
    async def generate_response(self, messages: List[BaseMessage]) -> AIMessage:
        """Generate response using OpenAI."""
        try:
            if not self.llm:
                await self.initialize()
            
            response = await self.llm.ainvoke(messages)
            return response
            
        except Exception as e:
            logger.error(f"OpenAI response generation failed: {str(e)}")
            raise LLMProviderError(f"OpenAI response generation failed: {str(e)}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            if not self.embeddings:
                await self.initialize()
            
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {str(e)}")
            raise LLMProviderError(f"OpenAI embedding generation failed: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check OpenAI provider health."""
        try:
            test_messages = [HumanMessage(content="Health check")]
            await self.generate_response(test_messages)
            return True
        except Exception:
            return False


class OllamaProvider(BaseLLMProvider):
    """Ollama provider implementation for local models."""
    
    def __init__(self, model_profile: ModelProfile):
        """Initialize Ollama provider."""
        super().__init__(model_profile)
        self.base_url = model_profile.api_base or "http://localhost:11434"
    
    async def initialize(self) -> None:
        """Initialize Ollama models."""
        try:
            # Initialize chat model
            self.llm = ChatOllama(
                model=self.model_profile.model_name,
                temperature=self.model_profile.temperature,
                base_url=self.base_url,
                **self.model_profile.additional_config
            )
            
            # Initialize embeddings model
            embedding_model = self.model_profile.additional_config.get(
                'embedding_model', 'nomic-embed-text'
            )
            self.embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=self.base_url
            )
            
            logger.info(f"Initialized Ollama provider with model: {self.model_profile.model_name}")
            
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Ollama provider: {str(e)}")
    
    async def generate_response(self, messages: List[BaseMessage]) -> AIMessage:
        """Generate response using Ollama."""
        try:
            if not self.llm:
                await self.initialize()
            
            response = await self.llm.ainvoke(messages)
            return response
            
        except Exception as e:
            logger.error(f"Ollama response generation failed: {str(e)}")
            raise LLMProviderError(f"Ollama response generation failed: {str(e)}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        try:
            if not self.embeddings:
                await self.initialize()
            
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
            
        except Exception as e:
            logger.error(f"Ollama embedding generation failed: {str(e)}")
            raise LLMProviderError(f"Ollama embedding generation failed: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check Ollama provider health."""
        try:
            test_messages = [HumanMessage(content="Health check")]
            await self.generate_response(test_messages)
            return True
        except Exception:
            return False


class OCIGenAIProvider(BaseLLMProvider):
    """Oracle Cloud Infrastructure Generative AI provider."""
    
    def __init__(self, model_profile: ModelProfile):
        """Initialize OCI GenAI provider."""
        super().__init__(model_profile)
        self.config_file = model_profile.additional_config.get('config_file')
        self.compartment_id = model_profile.additional_config.get('compartment_id')
        self.service_endpoint = model_profile.api_base
        
        if not self.compartment_id:
            raise LLMProviderError("OCI compartment_id is required")
    
    async def initialize(self) -> None:
        """Initialize OCI GenAI models."""
        try:
            # Try to import OCI SDK
            try:
                import oci
                from langchain_community.llms import OCIGenerativeAI
                from langchain_community.embeddings import OCIGenerativeAIEmbeddings
            except ImportError:
                raise LLMProviderError("OCI SDK not available. Install with: pip install oci")
            
            # Initialize configuration
            if self.config_file:
                config = oci.config.from_file(self.config_file)
            else:
                config = oci.config.from_file()  # Use default config
            
            # Initialize chat model
            self.llm = OCIGenerativeAI(
                model_id=self.model_profile.model_name,
                compartment_id=self.compartment_id,
                config=config,
                service_endpoint=self.service_endpoint,
                **self.model_profile.additional_config
            )
            
            # Initialize embeddings model
            embedding_model = self.model_profile.additional_config.get(
                'embedding_model', 'cohere.embed-english-v3.0'
            )
            self.embeddings = OCIGenerativeAIEmbeddings(
                model_id=embedding_model,
                compartment_id=self.compartment_id,
                config=config,
                service_endpoint=self.service_endpoint
            )
            
            logger.info(f"Initialized OCI GenAI provider with model: {self.model_profile.model_name}")
            
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize OCI GenAI provider: {str(e)}")
    
    async def generate_response(self, messages: List[BaseMessage]) -> AIMessage:
        """Generate response using OCI GenAI."""
        try:
            if not self.llm:
                await self.initialize()
            
            # Convert messages to text for OCI GenAI
            prompt = self._messages_to_prompt(messages)
            response_text = await self.llm.ainvoke(prompt)
            
            return AIMessage(content=response_text)
            
        except Exception as e:
            logger.error(f"OCI GenAI response generation failed: {str(e)}")
            raise LLMProviderError(f"OCI GenAI response generation failed: {str(e)}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OCI GenAI."""
        try:
            if not self.embeddings:
                await self.initialize()
            
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
            
        except Exception as e:
            logger.error(f"OCI GenAI embedding generation failed: {str(e)}")
            raise LLMProviderError(f"OCI GenAI embedding generation failed: {str(e)}")
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to a prompt string for OCI GenAI."""
        prompt_parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"Assistant: {message.content}")
        
        return "\n\n".join(prompt_parts)
    
    async def health_check(self) -> bool:
        """Check OCI GenAI provider health."""
        try:
            test_messages = [HumanMessage(content="Health check")]
            await self.generate_response(test_messages)
            return True
        except Exception:
            return False


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(model_profile: ModelProfile) -> BaseLLMProvider:
        """Create an LLM provider based on the model profile."""
        provider_map = {
            ModelProvider.OPENAI: OpenAIProvider,
            ModelProvider.LOCAL: OllamaProvider,  # Assuming LOCAL means Ollama
            # Add more providers as needed
        }
        
        # Special handling for OCI GenAI
        if model_profile.additional_config.get('provider') == 'oci_genai':
            return OCIGenAIProvider(model_profile)
        
        provider_class = provider_map.get(model_profile.provider)
        if not provider_class:
            raise LLMProviderError(f"Unsupported provider: {model_profile.provider}")
        
        return provider_class(model_profile)
    
    @staticmethod
    def create_llm_from_agent_spec(agent_spec: AgentSpec) -> BaseLLMProvider:
        """Create LLM provider from agent specification."""
        return LLMProviderFactory.create_provider(agent_spec.model_profile)


class LLMProviderManager:
    """Manager for multiple LLM providers."""
    
    def __init__(self):
        """Initialize the manager."""
        self.providers: Dict[str, BaseLLMProvider] = {}
    
    def register_provider(self, name: str, provider: BaseLLMProvider) -> None:
        """Register an LLM provider."""
        self.providers[name] = provider
        logger.info(f"Registered LLM provider: {name}")
    
    async def get_provider(self, name: str) -> BaseLLMProvider:
        """Get a provider by name."""
        if name not in self.providers:
            raise LLMProviderError(f"Provider '{name}' not found")
        
        provider = self.providers[name]
        if not provider.llm:
            await provider.initialize()
        
        return provider
    
    async def generate_response(self, provider_name: str, messages: List[BaseMessage]) -> AIMessage:
        """Generate response using specified provider."""
        provider = await self.get_provider(provider_name)
        return await provider.generate_response(messages)
    
    async def generate_embeddings(self, provider_name: str, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using specified provider."""
        provider = await self.get_provider(provider_name)
        return await provider.generate_embeddings(texts)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health_status = {}
        for name, provider in self.providers.items():
            try:
                health_status[name] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {name}: {str(e)}")
                health_status[name] = False
        
        return health_status


# Global LLM provider manager instance
llm_manager = LLMProviderManager()
