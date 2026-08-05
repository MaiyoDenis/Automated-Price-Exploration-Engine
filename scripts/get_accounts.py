import asyncio
import sys
import aiohttp
from loguru import logger
from project_apex.config.config import Config
from project_apex.config.environment import Environment

async def fetch_accounts():
    config = Config()
    env = Environment()
    
    rest_url = config.get_str("api", "rest_url")
    app_id = env.app_id
    token = env.deriv_token
    
    url = f"{rest_url}/trading/v1/options/accounts"
    headers = {
        "Authorization": f"Bearer {token}",
        "Deriv-App-ID": app_id
    }
    
    logger.info("Fetching your Deriv accounts...")
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                logger.error(f"Failed! HTTP {response.status}: {text}")
                return
                
            data = await response.json()
            logger.success("Successfully fetched accounts!")
            
            # Print accounts cleanly
            accounts = data.get("data", [])
            for i, acc in enumerate(accounts):
                logger.info(f"Account {i+1} Raw JSON: {acc}")

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(fetch_accounts())
