import asyncio
import sys
import aiohttp
from loguru import logger
from project_apex.config.config import Config
from project_apex.config.environment import Environment

async def test_otp():
    config = Config()
    env = Environment()
    
    rest_url = config.get_str("api", "rest_url")
    # For this test, if Account ID isn't set, we can't test. We'll try and see if it fails.
    try:
        account_id = env.account_id
    except ValueError:
        logger.error("DERIV_ACCOUNT_ID is not set in .env! Cannot proceed with OTP fetch.")
        return
        
    app_id = env.app_id
    token = env.deriv_token
    
    url = f"{rest_url}/trading/v1/options/accounts/{account_id}/otp"
    headers = {
        "Authorization": f"Bearer {token}",
        "Deriv-App-ID": app_id
    }
    
    logger.info(f"Fetching OTP from {url} with App ID {app_id}")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers) as response:
            logger.info(f"Status: {response.status}")
            try:
                data = await response.json()
                logger.info(f"Response JSON: {data}")
            except Exception as e:
                text = await response.text()
                logger.info(f"Response Text: {text}")

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(test_otp())
