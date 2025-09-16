# Reddit API Setup Guide

## Getting Reddit API Credentials

1. **Create Reddit Account** (if you don't have one)
   - Go to https://reddit.com and create an account

2. **Create Reddit App**
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Fill out the form:
     - **Name**: `agentic-venture-studio`
     - **App type**: Select "script"
     - **Description**: `Multi-agent system for discovering business opportunities`
     - **About URL**: Leave blank or use your GitHub repo
     - **Redirect URI**: `http://localhost:8000` (required but not used)

3. **Get Your Credentials**
   - After creating the app, you'll see:
     - **Client ID**: The string under your app name (looks like `abc123def456`)
     - **Client Secret**: The "secret" field (longer string)

## Configure Environment Variables

1. **Copy the template**:
   ```bash
   cp .env.template .env
   ```

2. **Edit `.env` file** with your credentials:
   ```bash
   REDDIT_CLIENT_ID=your_actual_client_id
   REDDIT_CLIENT_SECRET=your_actual_client_secret
   REDDIT_USER_AGENT=agentic-venture-studio/1.0 (by /u/yourusername)
   ```

3. **Customize subreddits** (optional):
   ```bash
   REDDIT_DEFAULT_SUBREDDITS=entrepreneur,startups,SaaS,YCombinator,business
   ```

## Rate Limiting

Reddit API has rate limits:
- **60 requests per minute** for authenticated requests
- **10 requests per minute** for unauthenticated requests

Our implementation includes automatic rate limiting and retry logic.

## Testing Your Setup

```bash
# Install dependencies
source venv/bin/activate
pip install aiohttp python-dotenv

# Test Reddit connection
python -c "
import asyncio
from src.agents.reddit_signals_scout import RedditSignalsScout
async def test():
    scout = RedditSignalsScout()
    await scout.start()
    signals = await scout.discover_signals(['entrepreneur'], timeframe='day')
    print(f'Found {len(signals)} signals')
    await scout.stop()
asyncio.run(test())
"
```

## Troubleshooting

**"Invalid Credentials" Error**:
- Double-check your Client ID and Secret
- Make sure there are no extra spaces
- Verify your User-Agent string is descriptive

**"Rate Limited" Error**:
- Wait a minute and try again
- Reduce the number of subreddits
- Check if you're making too many requests

**"Subreddit Not Found" Error**:
- Verify subreddit names are spelled correctly
- Some subreddits may be private or restricted