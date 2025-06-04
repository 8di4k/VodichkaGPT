# VodichkaGPT 🤖

An advanced Telegram bot powered by OpenAI's GPT models with multiple chat modes, image generation, voice message transcription, OCR capabilities, and subscription management.

## ✨ Features

### 🧠 Multiple Chat Modes
- **🟣 VodichkaGPT** - General AI assistant
- **👩🏼‍💻 Code Assistant** - Programming help and code generation
- **👩‍🎨 Artist** - AI image generation using DALL-E
- **📚 Homework Helper** - Educational assistance
- **👨‍⚖️ Lawyer** - Legal guidance and information
- **💻 Tech Support** - Technology troubleshooting
- **📝 Text Improver** - Grammar correction and text enhancement
- **🧠 Psychologist** - Emotional support and guidance
- **🚀 Elon Musk** - Chat with AI persona of Elon Musk
- **🌟 Motivator** - Inspiration and motivation
- **💰 Money Maker** - Business and investment advice
- **📊 SQL Assistant** - Database and SQL help
- **🧳 Travel Guide** - Travel recommendations and information
- **🥒 Rick Sanchez** - Rick and Morty character persona
- **🧮 Accountant** - Financial and accounting assistance
- **🎬 Movie Expert** - Film recommendations and discussions
- **🍏 Nutritionist** - Diet and nutrition advice
- **🇬🇧 English Tutor** - English language learning
- **🌱 Life Coach** - Personal development guidance
- **👗 Fashion Advisor** - Style and fashion recommendations
- **🔨 DIY Expert** - Do-it-yourself project assistance
- **🎮 Gamer** - Video game discussions and tips
- **🏋️ Personal Trainer** - Fitness and workout guidance
- **💡 Startup Idea Generator** - Business idea brainstorming
- **💞 Relationship Advisor** - Relationship guidance and support

### 🎯 Core Capabilities
- **Text Generation** - Powered by multiple GPT models
- **Image Generation** - DALL-E integration for creating images from text
- **Voice Transcription** - Convert voice messages to text using Whisper
- **OCR** - Extract text from images using Tesseract
- **Group Chat Support** - Works in Telegram group chats
- **Message Streaming** - Real-time word-by-word response display
- **Subscription Management** - Paid subscription with trial period
- **Payment Processing** - YooKassa integration for payments

### 💳 Subscription System
- **Trial Period** - 1 day free trial for new users
- **Monthly Subscription** - 349 RUB/month
- **Balance Tracking** - Monitor remaining subscription days
- **Automatic Renewal** - Seamless payment processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- OpenAI API Key
- MongoDB instance

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/VodichkaGPT.git
cd VodichkaGPT
```

2. **Configure environment variables**
```bash
cp config/config.env.example config/config.env
# Edit config/config.env with your values
```

3. **Update configuration files**
Edit `config/config.yml` and add your tokens:
```yaml
telegram_token: "YOUR_TELEGRAM_BOT_TOKEN"
openai_api_key: "YOUR_OPENAI_API_KEY"
```

4. **Start with Docker Compose**
```bash
docker-compose up -d
```

5. **Or run locally**
```bash
pip install -r requirements.txt
python bot/bot.py
```

## ⚙️ Configuration

### Environment Variables (`config/config.env`)
```env
MONGODB_PORT=27017
MONGO_EXPRESS_PORT=8081
MONGODB_PATH=./mongodb
```

### Main Configuration (`config/config.yml`)
- `telegram_token` - Your Telegram bot token
- `openai_api_key` - Your OpenAI API key
- `allowed_telegram_usernames` - Restrict access to specific users (empty = open to all)
- `new_dialog_timeout` - New conversation timeout in seconds (default: 600)
- `image_size` - Generated image size: "256x256", "512x512", or "1024x1024"
- `enable_message_streaming` - Show responses word-by-word

### Chat Modes (`config/chat_modes.yml`)
Customize chat modes by editing the YAML file. Each mode includes:
- `name` - Display name
- `welcome_message` - Greeting message
- `prompt_start` - System prompt for the AI
- `parse_mode` - Message formatting (html/markdown)

### Models (`config/models.yml`)
Configure available AI models and their settings.

## 🎮 Usage

### Basic Commands
- `/start` - Start the bot
- `/help` - Show help message
- `/new` - Start a new conversation
- `/retry` - Regenerate the last response
- `/mode` - Switch between chat modes
- `/settings` - Show available models
- `/balance` - Check subscription balance
- `/subscribe` - Renew subscription

### Group Chat Usage
1. Add the bot to your group
2. Make it an admin (message reading permissions)
3. Mention the bot: `@VodichkaGPT_bot your question`
4. Or reply to the bot's messages

### Special Features
- **Voice Messages** - Send voice notes for transcription
- **Images** - Send images for OCR text extraction
- **Image Generation** - Use Artist mode to create images from text descriptions

## 🏗️ Architecture

### Project Structure
```
VodichkaGPT/
├── bot/
│   ├── bot.py              # Main bot logic
│   ├── config.py           # Configuration loader
│   ├── database.py         # Database operations
│   └── openai_utils.py     # OpenAI API utilities
├── config/
│   ├── config.yml          # Main configuration
│   ├── config.env          # Environment variables
│   ├── chat_modes.yml      # Chat mode definitions
│   └── models.yml          # AI model configurations
├── docker-compose.yml      # Docker deployment
├── Dockerfile             # Container configuration
└── requirements.txt       # Python dependencies
```

### Technology Stack
- **Backend** - Python 3.8+
- **Bot Framework** - python-telegram-bot
- **AI Models** - OpenAI GPT-3.5/GPT-4
- **Database** - MongoDB
- **Deployment** - Docker + Docker Compose
- **Payments** - YooKassa
- **OCR** - Tesseract
- **Voice** - OpenAI Whisper

## 🔧 Development

### Local Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up MongoDB locally
docker run -d -p 27017:27017 --name mongo mongo:latest

# Run the bot
python bot/bot.py
```

### Adding New Chat Modes
1. Edit `config/chat_modes.yml`
2. Add your new mode configuration
3. Restart the bot

### Database Schema
The bot uses MongoDB with collections for:
- Users and their settings
- Chat histories and dialogs  
- Subscription information
- Usage statistics

## 📊 Monitoring

Access MongoDB admin interface at `http://localhost:8081` (Docker setup)
- Username: `your-username`
- Password: `your-password`

## 🔒 Security

- API keys are stored in configuration files (not in code)
- User access can be restricted via `allowed_telegram_usernames`
- Subscription validation prevents unauthorized usage
- Rate limiting prevents API abuse

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is proprietary software. All rights reserved.

## 📞 Support

For support and questions:
- Telegram Channel: [@VodichkaGPT](https://t.me/VodichkaGPT)
- Issues: Create an issue in this repository

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Custom AI model fine-tuning
- [ ] Enhanced group chat features
- [ ] Mobile app companion
- [ ] API for third-party integrations

---

**Made with ❤️ by the VodichkaGPT Team** 