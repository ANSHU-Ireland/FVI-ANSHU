import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from abc import ABC, abstractmethod
import yfinance as yf
import quandl
from alpha_vantage.timeseries import TimeSeries
import time

from ..config import settings

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
    
    @abstractmethod
    async def fetch_data(self, **kwargs) -> Dict[str, Any]:
        """Fetch data from the source."""
        pass
    
    def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()


class IEADataSource(DataSource):
    """International Energy Agency data source."""
    
    def __init__(self):
        super().__init__("IEA", settings.IEA_API_KEY)
        self.base_url = "https://api.iea.org/v1"
        self.rate_limit_delay = 1.0
    
    async def fetch_data(self, dataset: str, country: str = "WORLD", **kwargs) -> Dict[str, Any]:
        """Fetch IEA data."""
        self._rate_limit()
        
        if not self.api_key:
            logger.warning("IEA API key not configured")
            return {"error": "API key not configured"}
        
        try:
            url = f"{self.base_url}/stats/{dataset}"
            params = {
                "country": country,
                "apikey": self.api_key,
                **kwargs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_iea_data(data)
                    else:
                        logger.error(f"IEA API error: {response.status}")
                        return {"error": f"API error: {response.status}"}
        
        except Exception as e:
            logger.error(f"Error fetching IEA data: {e}")
            return {"error": str(e)}
    
    def _process_iea_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process IEA data response."""
        try:
            if "data" in raw_data:
                df = pd.DataFrame(raw_data["data"])
                return {
                    "source": "IEA",
                    "data": df.to_dict("records"),
                    "metadata": raw_data.get("metadata", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {"error": "No data in response"}
        except Exception as e:
            logger.error(f"Error processing IEA data: {e}")
            return {"error": str(e)}


class EIADataSource(DataSource):
    """U.S. Energy Information Administration data source."""
    
    def __init__(self):
        super().__init__("EIA", settings.EIA_API_KEY)
        self.base_url = "https://api.eia.gov/v2"
        self.rate_limit_delay = 1.0
    
    async def fetch_data(self, series_id: str, **kwargs) -> Dict[str, Any]:
        """Fetch EIA data."""
        self._rate_limit()
        
        if not self.api_key:
            logger.warning("EIA API key not configured")
            return {"error": "API key not configured"}
        
        try:
            url = f"{self.base_url}/seriesid/{series_id}"
            params = {
                "api_key": self.api_key,
                **kwargs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_eia_data(data)
                    else:
                        logger.error(f"EIA API error: {response.status}")
                        return {"error": f"API error: {response.status}"}
        
        except Exception as e:
            logger.error(f"Error fetching EIA data: {e}")
            return {"error": str(e)}
    
    def _process_eia_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process EIA data response."""
        try:
            if "response" in raw_data and "data" in raw_data["response"]:
                df = pd.DataFrame(raw_data["response"]["data"])
                return {
                    "source": "EIA",
                    "data": df.to_dict("records"),
                    "metadata": raw_data["response"].get("metadata", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {"error": "No data in response"}
        except Exception as e:
            logger.error(f"Error processing EIA data: {e}")
            return {"error": str(e)}


class QuandlDataSource(DataSource):
    """Quandl data source."""
    
    def __init__(self):
        super().__init__("Quandl", settings.QUANDL_API_KEY)
        self.rate_limit_delay = 0.5
        if self.api_key:
            quandl.ApiConfig.api_key = self.api_key
    
    async def fetch_data(self, database_code: str, dataset_code: str, **kwargs) -> Dict[str, Any]:
        """Fetch Quandl data."""
        self._rate_limit()
        
        if not self.api_key:
            logger.warning("Quandl API key not configured")
            return {"error": "API key not configured"}
        
        try:
            # Run in thread pool since quandl is synchronous
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None, 
                lambda: quandl.get(f"{database_code}/{dataset_code}", **kwargs)
            )
            
            return {
                "source": "Quandl",
                "data": data.reset_index().to_dict("records"),
                "metadata": {
                    "database": database_code,
                    "dataset": dataset_code,
                    "columns": data.columns.tolist()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error fetching Quandl data: {e}")
            return {"error": str(e)}


class AlphaVantageDataSource(DataSource):
    """Alpha Vantage data source."""
    
    def __init__(self):
        super().__init__("Alpha Vantage", settings.ALPHA_VANTAGE_API_KEY)
        self.rate_limit_delay = 12.0  # Free tier: 5 requests per minute
        if self.api_key:
            self.ts = TimeSeries(key=self.api_key, output_format='pandas')
    
    async def fetch_data(self, symbol: str, function: str = "TIME_SERIES_DAILY", **kwargs) -> Dict[str, Any]:
        """Fetch Alpha Vantage data."""
        self._rate_limit()
        
        if not self.api_key:
            logger.warning("Alpha Vantage API key not configured")
            return {"error": "API key not configured"}
        
        try:
            loop = asyncio.get_event_loop()
            
            if function == "TIME_SERIES_DAILY":
                data, meta_data = await loop.run_in_executor(
                    None, 
                    lambda: self.ts.get_daily(symbol=symbol, outputsize=kwargs.get('outputsize', 'compact'))
                )
            else:
                # Add more functions as needed
                return {"error": f"Function {function} not implemented"}
            
            return {
                "source": "Alpha Vantage",
                "data": data.reset_index().to_dict("records"),
                "metadata": meta_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            return {"error": str(e)}


class YFinanceDataSource(DataSource):
    """Yahoo Finance data source."""
    
    def __init__(self):
        super().__init__("Yahoo Finance")
        self.rate_limit_delay = 0.5
    
    async def fetch_data(self, symbol: str, period: str = "1y", **kwargs) -> Dict[str, Any]:
        """Fetch Yahoo Finance data."""
        self._rate_limit()
        
        try:
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist_data = await loop.run_in_executor(
                None, 
                lambda: ticker.history(period=period)
            )
            
            # Get info
            info = await loop.run_in_executor(
                None, 
                lambda: ticker.info
            )
            
            return {
                "source": "Yahoo Finance",
                "data": hist_data.reset_index().to_dict("records"),
                "metadata": info,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return {"error": str(e)}


class WorldBankDataSource(DataSource):
    """World Bank data source."""
    
    def __init__(self):
        super().__init__("World Bank")
        self.base_url = "https://api.worldbank.org/v2"
        self.rate_limit_delay = 1.0
    
    async def fetch_data(self, indicator: str, country: str = "all", **kwargs) -> Dict[str, Any]:
        """Fetch World Bank data."""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/country/{country}/indicator/{indicator}"
            params = {
                "format": "json",
                "per_page": kwargs.get("per_page", 1000),
                "date": kwargs.get("date", "2010:2023")
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_worldbank_data(data)
                    else:
                        logger.error(f"World Bank API error: {response.status}")
                        return {"error": f"API error: {response.status}"}
        
        except Exception as e:
            logger.error(f"Error fetching World Bank data: {e}")
            return {"error": str(e)}
    
    def _process_worldbank_data(self, raw_data: List[Any]) -> Dict[str, Any]:
        """Process World Bank data response."""
        try:
            if len(raw_data) >= 2 and isinstance(raw_data[1], list):
                df = pd.DataFrame(raw_data[1])
                return {
                    "source": "World Bank",
                    "data": df.to_dict("records"),
                    "metadata": raw_data[0] if len(raw_data) > 0 else {},
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {"error": "Invalid data format"}
        except Exception as e:
            logger.error(f"Error processing World Bank data: {e}")
            return {"error": str(e)}


class DataSourceManager:
    """Manage multiple data sources."""
    
    def __init__(self):
        self.sources = {
            "IEA": IEADataSource(),
            "EIA": EIADataSource(),
            "Quandl": QuandlDataSource(),
            "Alpha Vantage": AlphaVantageDataSource(),
            "Yahoo Finance": YFinanceDataSource(),
            "World Bank": WorldBankDataSource()
        }
    
    async def fetch_from_source(self, source_name: str, **kwargs) -> Dict[str, Any]:
        """Fetch data from a specific source."""
        if source_name not in self.sources:
            return {"error": f"Unknown data source: {source_name}"}
        
        source = self.sources[source_name]
        return await source.fetch_data(**kwargs)
    
    async def fetch_multiple_sources(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fetch data from multiple sources concurrently."""
        tasks = []
        for request in requests:
            source_name = request.pop("source", "")
            if source_name in self.sources:
                task = self.fetch_from_source(source_name, **request)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(results)
        }
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        return list(self.sources.keys())
    
    def get_source_info(self, source_name: str) -> Dict[str, Any]:
        """Get information about a specific source."""
        if source_name not in self.sources:
            return {"error": f"Unknown data source: {source_name}"}
        
        source = self.sources[source_name]
        return {
            "name": source.name,
            "api_key_configured": source.api_key is not None,
            "rate_limit_delay": source.rate_limit_delay
        }


# Mock data generator for testing
class MockDataGenerator:
    """Generate mock data for testing purposes."""
    
    @staticmethod
    def generate_coal_metrics_data(
        sub_industry: str = "Coal Mining",
        country: str = "Global",
        years: List[int] = None
    ) -> Dict[str, Any]:
        """Generate mock coal metrics data."""
        if years is None:
            years = list(range(2020, 2024))
        
        np.random.seed(42)  # For reproducible results
        
        metrics = {
            "necessity_score": np.random.uniform(60, 80, len(years)),
            "resource_scarcity_score": np.random.uniform(40, 70, len(years)),
            "artificial_support_score": np.random.uniform(30, 60, len(years)),
            "emissions_score": np.random.uniform(10, 30, len(years)),  # Lower is worse for coal
            "ecological_score": np.random.uniform(15, 35, len(years)),
            "workforce_transition_score": np.random.uniform(25, 50, len(years)),
            "infrastructure_repurposing_score": np.random.uniform(20, 45, len(years)),
            "monopoly_control_score": np.random.uniform(40, 70, len(years)),
            "economic_score": np.random.uniform(35, 65, len(years)),
            "technological_disruption_score": np.random.uniform(20, 40, len(years))
        }
        
        # Calculate composite FVI score
        weights = {
            "necessity_score": 0.15,
            "resource_scarcity_score": 0.10,
            "artificial_support_score": 0.10,
            "emissions_score": 0.15,
            "ecological_score": 0.10,
            "workforce_transition_score": 0.10,
            "infrastructure_repurposing_score": 0.10,
            "monopoly_control_score": 0.05,
            "economic_score": 0.10,
            "technological_disruption_score": 0.05
        }
        
        fvi_scores = []
        for i in range(len(years)):
            score = sum(metrics[metric][i] * weight for metric, weight in weights.items())
            fvi_scores.append(score)
        
        return {
            "sub_industry": sub_industry,
            "country": country,
            "years": years,
            "metrics": metrics,
            "fvi_scores": fvi_scores,
            "weights": weights,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def generate_scenario_data(base_data: Dict[str, Any], scenario_changes: Dict[str, float]) -> Dict[str, Any]:
        """Generate scenario data based on changes."""
        scenario_data = base_data.copy()
        scenario_metrics = scenario_data["metrics"].copy()
        
        # Apply scenario changes
        for metric, change in scenario_changes.items():
            if metric in scenario_metrics:
                scenario_metrics[metric] = [
                    max(0, min(100, value + change)) for value in scenario_metrics[metric]
                ]
        
        # Recalculate FVI scores
        weights = scenario_data["weights"]
        fvi_scores = []
        for i in range(len(scenario_data["years"])):
            score = sum(scenario_metrics[metric][i] * weight for metric, weight in weights.items())
            fvi_scores.append(score)
        
        scenario_data["metrics"] = scenario_metrics
        scenario_data["fvi_scores"] = fvi_scores
        scenario_data["scenario_changes"] = scenario_changes
        
        return scenario_data


# Export classes
__all__ = [
    "DataSource", "IEADataSource", "EIADataSource", "QuandlDataSource",
    "AlphaVantageDataSource", "YFinanceDataSource", "WorldBankDataSource",
    "DataSourceManager", "MockDataGenerator"
]
