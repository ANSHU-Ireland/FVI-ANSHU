"""
Global Energy Monitor Scraper
============================

Scrapy-Playwright spider for Global Energy Monitor's mine tracker.
Handles login flow and incremental diff with state persistence.
"""

import scrapy
from scrapy_playwright.page import PageMethod
from typing import Generator, Dict, Any
import json
import hashlib
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GEMSpider(scrapy.Spider):
    """Spider for Global Energy Monitor mine tracker."""
    
    name = 'gem_mines'
    allowed_domains = ['globalenergymonitor.org']
    start_urls = ['https://globalenergymonitor.org/projects/global-coal-mine-tracker/']
    
    # State file for incremental scraping
    state_file = Path('data/gem_state.json')
    
    custom_settings = {
        'DOWNLOAD_HANDLERS': {
            'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
        },
        'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            'headless': True,
            'timeout': 30000,
        },
        'CONCURRENT_REQUESTS': 1,  # Be respectful
        'DOWNLOAD_DELAY': 2,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_state = self.load_state()
        self.current_mines = {}
        
    def load_state(self) -> Dict[str, Any]:
        """Load previous scraping state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        
        return {
            'last_run': None,
            'mine_hashes': {},
            'total_mines': 0
        }
        
    def save_state(self):
        """Save current scraping state."""
        state = {
            'last_run': datetime.utcnow().isoformat(),
            'mine_hashes': self.current_mines,
            'total_mines': len(self.current_mines)
        }
        
        self.state_file.parent.mkdir(exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
    def generate_content_hash(self, mine_data: Dict) -> str:
        """Generate hash for mine data to detect changes."""
        # Remove timestamp fields for hash calculation
        hashable_data = {k: v for k, v in mine_data.items() 
                        if k not in ['scraped_at', 'updated_at']}
        content = json.dumps(hashable_data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
        
    def start_requests(self):
        """Start requests with Playwright."""
        yield scrapy.Request(
            url=self.start_urls[0],
            callback=self.handle_login,
            meta={
                'playwright': True,
                'playwright_page_methods': [
                    PageMethod('wait_for_load_state', 'networkidle'),
                    PageMethod('wait_for_timeout', 3000),
                ]
            }
        )
        
    async def handle_login(self, response):
        """Handle login flow if required."""
        page = response.meta['playwright_page']
        
        # Check if login is required
        login_selectors = [
            'input[type="email"]',
            'input[name="username"]', 
            'a[href*="login"]',
            '.login-form'
        ]
        
        needs_login = False
        for selector in login_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    needs_login = True
                    break
            except:
                continue
                
        if needs_login:
            logger.info("Login required - implementing login flow")
            yield from self.perform_login(response)
        else:
            logger.info("No login required - proceeding to data extraction")
            yield from self.extract_mine_data(response)
            
    async def perform_login(self, response):
        """Perform login flow."""
        page = response.meta['playwright_page']
        
        try:
            # Look for email/username field
            email_field = await page.query_selector('input[type="email"], input[name="username"]')
            if email_field:
                await email_field.fill('your-email@example.com')  # Replace with actual credentials
                
            # Look for password field
            password_field = await page.query_selector('input[type="password"]')
            if password_field:
                await password_field.fill('your-password')  # Replace with actual credentials
                
            # Submit form
            submit_button = await page.query_selector('button[type="submit"], input[type="submit"]')
            if submit_button:
                await submit_button.click()
                await page.wait_for_load_state('networkidle')
                
            logger.info("Login completed")
            
        except Exception as e:
            logger.error(f"Login failed: {e}")
            
        # Proceed to data extraction
        yield from self.extract_mine_data(response)
        
    async def extract_mine_data(self, response):
        """Extract mine data from the tracker."""
        page = response.meta['playwright_page']
        
        try:
            # Navigate to the actual data page/API endpoint
            data_url = 'https://globalenergymonitor.org/wp-content/uploads/2024/07/Global-Coal-Mine-Tracker-July-2024.xlsx'
            
            # Try to find dynamic data loading
            await page.wait_for_timeout(5000)
            
            # Look for mine data in various formats
            mine_selectors = [
                '.mine-item',
                '.coal-mine',
                'tr[data-mine]',
                '.facility-row'
            ]
            
            mines_found = False
            for selector in mine_selectors:
                mine_elements = await page.query_selector_all(selector)
                if mine_elements:
                    mines_found = True
                    logger.info(f"Found {len(mine_elements)} mines with selector: {selector}")
                    
                    for element in mine_elements:
                        mine_data = await self.extract_mine_details(element)
                        if mine_data:
                            yield mine_data
                    break
                    
            if not mines_found:
                # Try to extract from JSON data or API calls
                yield from self.extract_from_api_calls(page)
                
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            
    async def extract_mine_details(self, element) -> Dict[str, Any]:
        """Extract details from a mine element."""
        try:
            mine_data = {
                'source': 'GlobalEnergyMonitor',
                'scraped_at': datetime.utcnow().isoformat(),
            }
            
            # Extract common fields
            field_mappings = {
                'name': ['.mine-name', '.facility-name', 'h3', 'h4'],
                'country': ['.country', '.location-country', '[data-country]'],
                'status': ['.status', '.mine-status', '[data-status]'],
                'capacity': ['.capacity', '.production-capacity', '[data-capacity]'],
                'operator': ['.operator', '.company', '[data-operator]'],
                'coal_type': ['.coal-type', '[data-coal-type]'],
                'coordinates': ['.coordinates', '[data-lat]', '[data-lng]'],
            }
            
            for field, selectors in field_mappings.items():
                for selector in selectors:
                    try:
                        field_element = await element.query_selector(selector)
                        if field_element:
                            text = await field_element.inner_text()
                            if text and text.strip():
                                mine_data[field] = text.strip()
                                break
                    except:
                        continue
                        
            # Generate content hash
            content_hash = self.generate_content_hash(mine_data)
            mine_data['content_hash'] = content_hash
            
            # Check if this is a new or updated mine
            mine_id = mine_data.get('name', '') + mine_data.get('country', '')
            previous_hash = self.previous_state['mine_hashes'].get(mine_id)
            
            if previous_hash != content_hash:
                mine_data['is_updated'] = previous_hash is not None
                mine_data['is_new'] = previous_hash is None
                self.current_mines[mine_id] = content_hash
                return mine_data
            else:
                # No changes, skip
                self.current_mines[mine_id] = content_hash
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract mine details: {e}")
            return None
            
    async def extract_from_api_calls(self, page):
        """Extract data from API calls or AJAX requests."""
        try:
            # Listen for network requests
            api_data = []
            
            def handle_response(response):
                if 'api' in response.url or 'json' in response.url:
                    try:
                        data = response.json()
                        api_data.append(data)
                    except:
                        pass
                        
            page.on('response', handle_response)
            
            # Trigger data loading
            await page.reload()
            await page.wait_for_timeout(10000)
            
            # Process collected API data
            for data in api_data:
                if isinstance(data, list):
                    for item in data:
                        yield self.process_api_mine_data(item)
                elif isinstance(data, dict) and 'mines' in data:
                    for mine in data['mines']:
                        yield self.process_api_mine_data(mine)
                        
        except Exception as e:
            logger.error(f"API extraction failed: {e}")
            
    def process_api_mine_data(self, mine_item: Dict) -> Dict[str, Any]:
        """Process mine data from API response."""
        mine_data = {
            'source': 'GlobalEnergyMonitor',
            'scraped_at': datetime.utcnow().isoformat(),
            'extraction_method': 'api',
            **mine_item  # Merge API data
        }
        
        # Generate content hash and check for changes
        content_hash = self.generate_content_hash(mine_data)
        mine_data['content_hash'] = content_hash
        
        mine_id = mine_data.get('name', '') + mine_data.get('country', '')
        previous_hash = self.previous_state['mine_hashes'].get(mine_id)
        
        if previous_hash != content_hash:
            mine_data['is_updated'] = previous_hash is not None
            mine_data['is_new'] = previous_hash is None
            self.current_mines[mine_id] = content_hash
            return mine_data
        else:
            self.current_mines[mine_id] = content_hash
            return None
            
    def closed(self, reason):
        """Called when spider closes."""
        self.save_state()
        
        # Log summary
        new_mines = sum(1 for mine_id, hash_val in self.current_mines.items() 
                       if mine_id not in self.previous_state['mine_hashes'])
        updated_mines = sum(1 for mine_id, hash_val in self.current_mines.items() 
                           if mine_id in self.previous_state['mine_hashes'] 
                           and self.previous_state['mine_hashes'][mine_id] != hash_val)
        
        logger.info(f"Scraping completed: {new_mines} new mines, {updated_mines} updated mines")


# Pipeline for processing scraped data
class GEMPipeline:
    """Pipeline for processing GEM mine data."""
    
    def __init__(self):
        self.output_file = Path('data/gem_mines.jsonl')
        self.file_handle = None
        
    def open_spider(self, spider):
        """Open output file."""
        self.output_file.parent.mkdir(exist_ok=True)
        self.file_handle = open(self.output_file, 'a', encoding='utf-8')
        
    def process_item(self, item, spider):
        """Process each mine item."""
        if item:  # Only process non-None items (changed mines)
            line = json.dumps(dict(item)) + '\n'
            self.file_handle.write(line)
            
        return item
        
    def close_spider(self, spider):
        """Close output file."""
        if self.file_handle:
            self.file_handle.close()


# Settings for the spider
CUSTOM_SETTINGS = {
    'ITEM_PIPELINES': {
        'gem_spider.GEMPipeline': 300,
    },
    'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 30000,
    'PLAYWRIGHT_DEFAULT_TIMEOUT': 30000,
}
