"""
Data Source Connectors
=====================

High-value structured feed connectors for IEA, EIA, World Bank, UN Comtrade, IMF.
Implements managed pulls with rate limiting, error handling, and incremental sync.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for a data source connector."""
    name: str
    base_url: str
    api_key_env: str
    rate_limit: int  # requests per minute
    timeout: int = 30
    incremental_field: Optional[str] = None
    trust_rank: int = 1  # 1=highest, 5=lowest


class ConnectorBase:
    """Base class for data source connectors."""
    
    def __init__(self, config: DataSourceConfig, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.last_request_time = None
        
    async def _rate_limit(self):
        """Enforce rate limiting."""
        if self.last_request_time:
            elapsed = datetime.now() - self.last_request_time
            min_interval = timedelta(minutes=1) / self.config.rate_limit
            if elapsed < min_interval:
                await asyncio.sleep((min_interval - elapsed).total_seconds())
        self.last_request_time = datetime.now()
        
    async def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Make rate-limited HTTP request."""
        await self._rate_limit()
        
        try:
            async with self.session.get(
                url, 
                params=params, 
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Request failed for {self.config.name}: {e}")
            raise
            
    def _generate_hash(self, data: Any) -> str:
        """Generate provenance hash for data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        
    async def extract(self, **kwargs) -> List[Dict]:
        """Extract data from source. Override in subclasses."""
        raise NotImplementedError


class IEAConnector(ConnectorBase):
    """International Energy Agency API connector."""
    
    def __init__(self, session: aiohttp.ClientSession):
        config = DataSourceConfig(
            name="IEA",
            base_url="https://api.iea.org",
            api_key_env="IEA_API_KEY",
            rate_limit=60,
            trust_rank=1
        )
        super().__init__(config, session)
        
    async def extract_coal_data(self, countries: List[str] = None) -> List[Dict]:
        """Extract coal production and consumption data."""
        countries = countries or ['WORLD', 'USA', 'CHN', 'IND', 'AUS', 'POL']
        
        datasets = [
            'COALINFO',  # Coal information
            'COALRESERVES',  # Coal reserves
            'COALPROD',  # Coal production
        ]
        
        results = []
        for dataset in datasets:
            for country in countries:
                try:
                    url = f"{self.config.base_url}/statistics/{dataset}"
                    params = {
                        'country': country,
                        'product': 'COAL',
                        'format': 'json'
                    }
                    
                    data = await self._make_request(url, params)
                    
                    record = {
                        'source': 'IEA',
                        'dataset': dataset,
                        'country': country,
                        'data': data,
                        'extracted_at': datetime.utcnow().isoformat(),
                        'provenance_hash': self._generate_hash(data),
                        'trust_rank': self.config.trust_rank
                    }
                    results.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract {dataset} for {country}: {e}")
                    
        logger.info(f"Extracted {len(results)} records from IEA")
        return results


class EIAConnector(ConnectorBase):
    """US Energy Information Administration connector."""
    
    def __init__(self, session: aiohttp.ClientSession):
        config = DataSourceConfig(
            name="EIA",
            base_url="https://api.eia.gov/v2",
            api_key_env="EIA_API_KEY", 
            rate_limit=1000,  # EIA has high rate limits
            trust_rank=1
        )
        super().__init__(config, session)
        
    async def extract_coal_data(self) -> List[Dict]:
        """Extract US coal data."""
        series_ids = [
            'COAL.CONS_TOT.A',  # Total coal consumption
            'COAL.PROD_TOT.A',  # Total coal production
            'COAL.STOCKS_TOT.M',  # Coal stocks
            'COAL.EXPORTS.A',  # Coal exports
            'COAL.IMPORTS.A',  # Coal imports
        ]
        
        results = []
        for series_id in series_ids:
            try:
                url = f"{self.config.base_url}/coal/consumption/data"
                params = {
                    'api_key': 'YOUR_API_KEY',  # Will be replaced with env var
                    'frequency': 'annual',
                    'data': [series_id],
                    'sort': [{'column': 'period', 'direction': 'desc'}],
                    'length': 50
                }
                
                data = await self._make_request(url, params)
                
                record = {
                    'source': 'EIA',
                    'series_id': series_id,
                    'data': data,
                    'extracted_at': datetime.utcnow().isoformat(),
                    'provenance_hash': self._generate_hash(data),
                    'trust_rank': self.config.trust_rank
                }
                results.append(record)
                
            except Exception as e:
                logger.warning(f"Failed to extract EIA series {series_id}: {e}")
                
        logger.info(f"Extracted {len(results)} records from EIA")
        return results


class WorldBankConnector(ConnectorBase):
    """World Bank CMAP (Climate Change Action Plan) connector."""
    
    def __init__(self, session: aiohttp.ClientSession):
        config = DataSourceConfig(
            name="WorldBank",
            base_url="https://api.worldbank.org/v2",
            api_key_env="",  # World Bank API is open
            rate_limit=100,
            trust_rank=1
        )
        super().__init__(config, session)
        
    async def extract_climate_data(self, countries: List[str] = None) -> List[Dict]:
        """Extract climate and energy indicators."""
        countries = countries or ['WLD', 'USA', 'CHN', 'IND', 'AUS', 'POL']
        
        indicators = [
            'EG.USE.COAL.KT.OE',  # Coal consumption
            'EN.ATM.CO2E.KT',     # CO2 emissions
            'NY.GDP.MKTP.CD',     # GDP
            'EG.ELC.COAL.ZS',     # Electricity from coal
        ]
        
        results = []
        for indicator in indicators:
            for country in countries:
                try:
                    url = f"{self.config.base_url}/country/{country}/indicator/{indicator}"
                    params = {
                        'format': 'json',
                        'per_page': 100,
                        'date': '2000:2024'
                    }
                    
                    data = await self._make_request(url, params)
                    
                    record = {
                        'source': 'WorldBank',
                        'indicator': indicator,
                        'country': country,
                        'data': data,
                        'extracted_at': datetime.utcnow().isoformat(),
                        'provenance_hash': self._generate_hash(data),
                        'trust_rank': self.config.trust_rank
                    }
                    results.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract {indicator} for {country}: {e}")
                    
        logger.info(f"Extracted {len(results)} records from World Bank")
        return results


class UNComtradeConnector(ConnectorBase):
    """UN Comtrade API connector for coal trade data."""
    
    def __init__(self, session: aiohttp.ClientSession):
        config = DataSourceConfig(
            name="UNComtrade",
            base_url="https://comtradeapi.un.org/data/v1/get",
            api_key_env="UN_COMTRADE_API_KEY",
            rate_limit=100,  # UN Comtrade has rate limits
            trust_rank=2
        )
        super().__init__(config, session)
        
    async def extract_coal_trade(self, years: List[int] = None) -> List[Dict]:
        """Extract coal trade data."""
        years = years or [2020, 2021, 2022, 2023, 2024]
        
        # Coal commodity codes (HS classification)
        coal_codes = [
            '2701',  # Coal; briquettes, ovoids and similar solid fuels manufactured from coal
            '270111', # Coal; anthracite, whether or not pulverised, but not agglomerated
            '270112', # Coal; bituminous, whether or not pulverised, but not agglomerated
        ]
        
        results = []
        for year in years:
            for code in coal_codes:
                try:
                    params = {
                        'typeCode': 'C',  # Commodities
                        'freqCode': 'A',  # Annual
                        'clCode': 'HS',   # Harmonized System
                        'period': str(year),
                        'reporterCode': 'all',
                        'cmdCode': code,
                        'flowCode': 'all',
                        'partnerCode': 'all',
                        'partner2Code': '',
                        'customsCode': '',
                        'motCode': '',
                        'maxRecords': 5000,
                        'format': 'json',
                        'includeDesc': 'true'
                    }
                    
                    data = await self._make_request(self.config.base_url, params)
                    
                    record = {
                        'source': 'UNComtrade',
                        'commodity_code': code,
                        'year': year,
                        'data': data,
                        'extracted_at': datetime.utcnow().isoformat(),
                        'provenance_hash': self._generate_hash(data),
                        'trust_rank': self.config.trust_rank
                    }
                    results.append(record)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract coal trade {code} for {year}: {e}")
                    
        logger.info(f"Extracted {len(results)} records from UN Comtrade")
        return results


class IMFSubsidyConnector(ConnectorBase):
    """IMF Energy Subsidy Database connector."""
    
    def __init__(self, session: aiohttp.ClientSession):
        config = DataSourceConfig(
            name="IMF",
            base_url="https://www.imf.org/external/datamapper/api/v1",
            api_key_env="",  # IMF API is open
            rate_limit=60,
            trust_rank=1
        )
        super().__init__(config, session)
        
    async def extract_subsidy_data(self) -> List[Dict]:
        """Extract energy subsidy data."""
        indicators = [
            'ENDA',  # Energy subsidies
            'NGDP_RPCH',  # Real GDP growth
            'PCPIPCH',    # Inflation
        ]
        
        results = []
        for indicator in indicators:
            try:
                url = f"{self.config.base_url}/{indicator}"
                
                data = await self._make_request(url)
                
                record = {
                    'source': 'IMF',
                    'indicator': indicator,
                    'data': data,
                    'extracted_at': datetime.utcnow().isoformat(),
                    'provenance_hash': self._generate_hash(data),
                    'trust_rank': self.config.trust_rank
                }
                results.append(record)
                
            except Exception as e:
                logger.warning(f"Failed to extract IMF indicator {indicator}: {e}")
                
        logger.info(f"Extracted {len(results)} records from IMF")
        return results


class ConnectorOrchestrator:
    """Orchestrates all data source connectors."""
    
    def __init__(self):
        self.connectors = []
        
    async def initialize(self):
        """Initialize all connectors with shared session."""
        self.session = aiohttp.ClientSession()
        
        self.connectors = [
            IEAConnector(self.session),
            EIAConnector(self.session),
            WorldBankConnector(self.session),
            UNComtradeConnector(self.session),
            IMFSubsidyConnector(self.session),
        ]
        
    async def extract_all(self) -> Dict[str, List[Dict]]:
        """Extract data from all sources."""
        results = {}
        
        for connector in self.connectors:
            try:
                source_name = connector.config.name
                logger.info(f"Extracting from {source_name}")
                
                if isinstance(connector, IEAConnector):
                    data = await connector.extract_coal_data()
                elif isinstance(connector, EIAConnector):
                    data = await connector.extract_coal_data()
                elif isinstance(connector, WorldBankConnector):
                    data = await connector.extract_climate_data()
                elif isinstance(connector, UNComtradeConnector):
                    data = await connector.extract_coal_trade()
                elif isinstance(connector, IMFSubsidyConnector):
                    data = await connector.extract_subsidy_data()
                else:
                    data = await connector.extract()
                    
                results[source_name] = data
                
            except Exception as e:
                logger.error(f"Failed to extract from {connector.config.name}: {e}")
                results[connector.config.name] = []
                
        return results
        
    async def close(self):
        """Close the session."""
        if hasattr(self, 'session'):
            await self.session.close()


async def main():
    """Main function for testing connectors."""
    orchestrator = ConnectorOrchestrator()
    
    try:
        await orchestrator.initialize()
        results = await orchestrator.extract_all()
        
        # Print summary
        for source, data in results.items():
            print(f"{source}: {len(data)} records extracted")
            
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(main())
