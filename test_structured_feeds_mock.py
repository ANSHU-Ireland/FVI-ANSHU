"""
Test version of structured feeds connector with mock data
=========================================================

This version demonstrates the connector functionality with mock data
when real API endpoints are not available or require authentication.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from src.connectors.structured_feeds import ConnectorBase, DataSourceConfig

logger = logging.getLogger(__name__)


class MockConnectorBase(ConnectorBase):
    """Base class for mock data connectors."""
    
    async def _make_request(self, url: str, params: Dict = None) -> Dict:
        """Override to return mock data instead of making real requests."""
        await self._rate_limit()
        
        # Simulate API response time
        await asyncio.sleep(0.1)
        
        # Return mock data based on the URL
        if 'iea.org' in url:
            return self._get_mock_iea_data()
        elif 'eia.gov' in url:
            return self._get_mock_eia_data()
        elif 'worldbank.org' in url:
            return self._get_mock_worldbank_data()
        elif 'comtradeapi.un.org' in url:
            return self._get_mock_comtrade_data()
        elif 'imf.org' in url:
            return self._get_mock_imf_data()
        else:
            return {'mock_data': True, 'url': url, 'params': params}
    
    def _get_mock_iea_data(self) -> Dict:
        """Mock IEA data."""
        return {
            'data': [
                {
                    'country': 'WORLD',
                    'product': 'COAL',
                    'year': 2023,
                    'value': 8500.5,
                    'unit': 'Million tonnes'
                },
                {
                    'country': 'USA',
                    'product': 'COAL',
                    'year': 2023,
                    'value': 650.2,
                    'unit': 'Million tonnes'
                }
            ],
            'status': 'success'
        }
    
    def _get_mock_eia_data(self) -> Dict:
        """Mock EIA data."""
        return {
            'response': {
                'data': [
                    {
                        'period': '2023',
                        'seriesId': 'COAL.CONS_TOT.A',
                        'value': 535.5,
                        'units': 'Million short tons'
                    },
                    {
                        'period': '2023',
                        'seriesId': 'COAL.PROD_TOT.A',
                        'value': 595.4,
                        'units': 'Million short tons'
                    }
                ]
            }
        }
    
    def _get_mock_worldbank_data(self) -> Dict:
        """Mock World Bank data."""
        return [
            None,  # World Bank API returns metadata as first element
            [
                {
                    'indicator': {'id': 'EG.USE.COAL.KT.OE', 'value': 'Coal consumption'},
                    'country': {'id': 'USA', 'value': 'United States'},
                    'countryiso3code': 'USA',
                    'date': '2023',
                    'value': 425.8,
                    'unit': 'kt of oil equivalent'
                },
                {
                    'indicator': {'id': 'EN.ATM.CO2E.KT', 'value': 'CO2 emissions'},
                    'country': {'id': 'USA', 'value': 'United States'},
                    'countryiso3code': 'USA',
                    'date': '2023',
                    'value': 5200000,
                    'unit': 'kt'
                }
            ]
        ]
    
    def _get_mock_comtrade_data(self) -> Dict:
        """Mock UN Comtrade data."""
        return {
            'data': [
                {
                    'period': '2023',
                    'reporterCode': 'USA',
                    'reporterDesc': 'United States',
                    'partnerCode': 'CHN',
                    'partnerDesc': 'China',
                    'cmdCode': '2701',
                    'cmdDesc': 'Coal',
                    'flowCode': 'X',
                    'flowDesc': 'Export',
                    'tradeValue': 1250000,
                    'netWeight': 8500
                }
            ]
        }
    
    def _get_mock_imf_data(self) -> Dict:
        """Mock IMF data."""
        return {
            'values': {
                'ENDA': {
                    'USA': [
                        {'year': 2023, 'value': 15.2}
                    ],
                    'CHN': [
                        {'year': 2023, 'value': 42.8}
                    ]
                }
            }
        }


class MockIEAConnector(MockConnectorBase):
    """Mock IEA connector with test data."""
    
    def __init__(self, session: aiohttp.ClientSession):
        config = DataSourceConfig(
            name="IEA_MOCK",
            base_url="https://api.iea.org",
            api_key_env="IEA_API_KEY",
            rate_limit=60,
            trust_rank=1
        )
        super().__init__(config, session)
        
    async def extract_coal_data(self, countries: List[str] = None) -> List[Dict]:
        """Extract mock coal data."""
        countries = countries or ['WORLD', 'USA', 'CHN']
        datasets = ['COALINFO', 'COALRESERVES', 'COALPROD']
        
        results = []
        for dataset in datasets:
            for country in countries:
                url = f"{self.config.base_url}/statistics/{dataset}"
                params = {'country': country, 'product': 'COAL', 'format': 'json'}
                
                data = await self._make_request(url, params)
                
                record = {
                    'source': 'IEA_MOCK',
                    'dataset': dataset,
                    'country': country,
                    'data': data,
                    'extracted_at': datetime.utcnow().isoformat(),
                    'provenance_hash': self._generate_hash(data),
                    'trust_rank': self.config.trust_rank
                }
                results.append(record)
                
        logger.info(f"Extracted {len(results)} mock records from IEA")
        return results


class MockEIAConnector(MockConnectorBase):
    """Mock EIA connector with test data."""
    
    def __init__(self, session: aiohttp.ClientSession):
        config = DataSourceConfig(
            name="EIA_MOCK",
            base_url="https://api.eia.gov/v2",
            api_key_env="EIA_API_KEY",
            rate_limit=1000,
            trust_rank=1
        )
        super().__init__(config, session)
        
    async def extract_coal_data(self) -> List[Dict]:
        """Extract mock US coal data."""
        series_ids = ['COAL.CONS_TOT.A', 'COAL.PROD_TOT.A', 'COAL.EXPORTS.A']
        
        results = []
        for series_id in series_ids:
            url = f"{self.config.base_url}/coal/consumption/data"
            params = {'series_id': series_id}
            
            data = await self._make_request(url, params)
            
            record = {
                'source': 'EIA_MOCK',
                'series_id': series_id,
                'data': data,
                'extracted_at': datetime.utcnow().isoformat(),
                'provenance_hash': self._generate_hash(data),
                'trust_rank': self.config.trust_rank
            }
            results.append(record)
            
        logger.info(f"Extracted {len(results)} mock records from EIA")
        return results


class MockConnectorOrchestrator:
    """Mock orchestrator for testing purposes."""
    
    def __init__(self):
        self.connectors = []
        
    async def initialize(self):
        """Initialize mock connectors."""
        self.session = aiohttp.ClientSession()
        
        self.connectors = [
            MockIEAConnector(self.session),
            MockEIAConnector(self.session),
        ]
        
    async def extract_all(self) -> Dict[str, List[Dict]]:
        """Extract mock data from all sources."""
        results = {}
        
        for connector in self.connectors:
            try:
                source_name = connector.config.name
                logger.info(f"Extracting mock data from {source_name}")
                
                if isinstance(connector, MockIEAConnector):
                    data = await connector.extract_coal_data()
                elif isinstance(connector, MockEIAConnector):
                    data = await connector.extract_coal_data()
                else:
                    data = []
                    
                results[source_name] = data
                
            except Exception as e:
                logger.error(f"Failed to extract from {connector.config.name}: {e}")
                results[connector.config.name] = []
                
        return results
        
    async def close(self):
        """Close the session."""
        if hasattr(self, 'session'):
            await self.session.close()


async def test_mock_connectors():
    """Test the mock connectors."""
    print("🧪 Testing Mock Structured Feeds Connectors")
    print("=" * 50)
    
    orchestrator = MockConnectorOrchestrator()
    
    try:
        await orchestrator.initialize()
        results = await orchestrator.extract_all()
        
        print("\n📊 Mock Extraction Results:")
        total_records = 0
        for source, data in results.items():
            record_count = len(data)
            total_records += record_count
            print(f"   {source}: {record_count} records")
            
            # Show sample data
            if data:
                sample = data[0]
                print(f"   Sample: {sample['dataset'] if 'dataset' in sample else sample.get('series_id', 'N/A')}")
        
        print(f"\n✅ Total Records Extracted: {total_records}")
        
        # Show detailed data for first source
        if results:
            first_source = list(results.keys())[0]
            first_data = results[first_source]
            if first_data:
                print(f"\n📄 Sample Record from {first_source}:")
                sample_record = first_data[0]
                print(f"   Dataset: {sample_record.get('dataset', 'N/A')}")
                print(f"   Country: {sample_record.get('country', 'N/A')}")
                print(f"   Extracted: {sample_record.get('extracted_at', 'N/A')}")
                print(f"   Hash: {sample_record.get('provenance_hash', 'N/A')}")
                print(f"   Trust Rank: {sample_record.get('trust_rank', 'N/A')}")
                
    except Exception as e:
        print(f"❌ Error during mock extraction: {e}")
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_mock_connectors())
