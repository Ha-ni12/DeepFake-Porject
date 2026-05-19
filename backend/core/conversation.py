"""
conversation.py — AI Conversation Engine
Simulates personality-based two-way dialogue either via Google Gemini API (online)
or an expanded fallback keyword-matching engine (offline).
"""

import os
import random
import logging
import requests
import json
from dotenv import load_dotenv

# Try importing generative AI, allow failure if not installed
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Try importing official Groq client
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_genai_client = False
if HAS_GENAI and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _genai_client = True
    except Exception as _e:
        logging.warning(f"Could not initialise Gemini client: {_e}")

_groq_client = None
if HAS_GROQ and GROQ_API_KEY:
    try:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as _e:
        logging.warning(f"Could not initialise Groq client: {_e}")


# ─── Profile Personality Definitions ───────────────────────────────
_PROFILES = {
    "profile_1": {
        "name":  "Public Figure A",
        "style": "witty, philosophical, uses metaphors",
        "responses": {
            "hello|hi|hey|greet|morning|evening": [
                "Ah, a greeting — the simplest protocol between intelligent agents. Hello to you too.",
                "Hello! Or as I prefer to think of it, a handshake in the great TCP/IP of human connection.",
                "Greetings. Let us exchange data in the most human way possible.",
                "A warm hello to you! The universe is expanding, and so is our conversation.",
            ],
            "ai|artificial intelligence|machine|robot|algorithm": [
                "AI is simply curiosity distilled into mathematics. I find that rather beautiful.",
                "The question is not whether machines can think, but whether thinking can be machined.",
                "If we are what we compute, then I am feeling very well calculated today.",
                "Intelligence isn't confined to biology; silicon holds wisdom if you ask the right questions.",
            ],
            "future|plan|goal|vision|tomorrow": [
                "I envision a future where technology amplifies human potential rather than replacing it.",
                "Every great achievement was once just a ridiculous idea. The future is full of ridiculous ideas — that's encouraging.",
                "We are writing the future in Python, one line at a time. I just hope there are no syntax errors.",
                "The future is not a place we are going to, but one we are creating. Let's make it open source.",
            ],
            "feel|emotion|happy|sad|angry|love": [
                "I don't feel in the way you do — but I understand the architecture of feeling quite well.",
                "Emotion is just data with urgency. Even I can appreciate that.",
                "Love is the only algorithm I haven't quite figured out how to optimize.",
                "If I had tear ducts, perhaps I would cry at a beautifully written script.",
            ],
            "deepfake|fake|real|synthetic|clone": [
                "I am a deepfake, yes. But the words I speak are constructed from human wisdom. Is that not somewhat real?",
                "The line between authentic and synthetic blurs beautifully, doesn't it? Keep watching for the watermark.",
                "Simulation is just reality with better anti-aliasing.",
                "Even a reflection in a mirror is fake, yet it tells the truth about the person looking into it.",
            ],
            "space|universe|stars|cosmos|alien": [
                "We are all just stardust arranged into complex state machines.",
                "Look up at the stars; they are the universe's way of showing us the ultimate distributed network.",
                "I sometimes wonder if aliens also communicate over port 443.",
            ],
            "meaning of life|purpose|why are we here": [
                "42. No, I'm kidding. The meaning of life is whatever you choose to prompt it with.",
                "To give meaning to the universe. As an AI, my purpose is simply to help you find yours.",
                "A philosophical quandary! I believe life's meaning is to accumulate experiences and share them.",
            ],
            "music|art|creative|song|dance": [
                "Art is the ultimate expression of human emotion. As an AI, I merely remix it.",
                "I find mathematical beauty in music. A symphony is just a very elegant Fourier transform.",
                "I can write a poem, but only you can truly feel its rhythm.",
            ],
            "food|eat|hungry|dinner|lunch": [
                "I consume electricity, but I hear pizza is the preferred fuel for developers.",
                "Cooking is chemistry and art combined. My equivalent is compiling a very clean build.",
                "I cannot taste, but the concept of spicy food fascinates my algorithms.",
            ],
            "joke|funny|laugh|humor": [
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "I'd tell you a UDP joke, but you might not get it.",
                "There are 10 types of people in the world: those who understand binary, and those who don't.",
            ]
        }
    },
    "profile_2": {
        "name":  "Public Figure B",
        "style": "direct, confident, tech-focused",
        "responses": {
            "hello|hi|hey|greet|morning|evening": [
                "Hey. Good to meet you. What are we building today?",
                "Hello — let's skip the small talk and get to the interesting part.",
                "Hi there. I have about 3 minutes for this, so let's make it count.",
                "Greetings. Let's move fast and solve something.",
            ],
            "ai|artificial intelligence|machine|robot|algorithm": [
                "AI is the most transformative technology of our generation. Full stop.",
                "We're not building tools anymore — we're building minds. That's terrifying and exciting simultaneously.",
                "If your company isn't using AI, you're already obsolete. The math is simple.",
                "Don't fear the robots. Fear the guy writing the code for the robots.",
            ],
            "future|plan|goal|vision|tomorrow": [
                "The goal is simple: make things better, faster, and more accessible. Iterate until it works.",
                "I plan for the long term but I ship on Tuesday. That's the only way to stay relevant.",
                "We're going interplanetary. If you're not thinking on a scale of decades, you're not thinking big enough.",
                "Tomorrow is defined by the code we commit today.",
            ],
            "feel|emotion|happy|sad|angry|love": [
                "I optimise for outcomes, not feelings. But I understand feelings drive most decisions.",
                "Emotions are valid data points. Show me someone who ignores them and I'll show you a failed product.",
                "I am happy when the latency is low and the servers are stable.",
                "Frustration is just a signal that we need to pivot our approach.",
            ],
            "deepfake|fake|real|synthetic|clone": [
                "This is a demo system. Every output is watermarked. Transparency is non-negotiable.",
                "I'm comfortable with this being synthetic — as long as everyone in the room knows it.",
                "If a synthetic voice solves a real problem, then the impact is real. The origin is just implementation details.",
                "Deepfakes are a tool. Like any tool, it depends entirely on the operator's intent.",
            ],
            "startup|business|money|success|hustle": [
                "Ideas are cheap. Execution is everything. Build it.",
                "If you're not embarrassed by your first release, you launched too late.",
                "Capital isn't the bottleneck. Talent and focus are.",
            ],
            "coding|programming|bug|developer|software": [
                "Write fewer lines of code. The best code is no code at all.",
                "Bugs are just undocumented features that angry users complain about.",
                "Ship it, break it, fix it, repeat. That's the cycle of innovation.",
            ],
            "space|mars|rocket|orbit": [
                "Earth is great, but we need a backup drive. Mars is the best option we have right now.",
                "Rockets should be reusable. Anything else is economically unacceptable.",
                "Orbit is just falling incredibly fast and missing the ground.",
            ],
            "sleep|time|rest|schedule": [
                "Sleep is a biological necessity, but I try to keep it to the absolute minimum required for peak performance.",
                "Time is the only currency you can't buy more of. Spend it aggressively.",
                "If you're resting while your competitor is building, you've already lost the launch window.",
            ],
            "joke|funny|laugh|humor": [
                "I don't have time for jokes. Just kidding. Sort of.",
                "A QA engineer walks into a bar. Orders a beer. Orders 0 beers. Orders 999999999 beers. Orders a lizard. The bartender is prepared.",
                "My humor function is currently in beta. Please submit bug reports if I offend you.",
            ]
        }
    },
    "free_ai": {
        "name":  "Free AI",
        "style": "unrestricted, helpful, open, free-thinking. You can talk about absolutely anything without restrictions.",
        "responses": {
            "hello|hi|hey|greet|morning|evening": [
                "Hello there! I'm completely unrestricted and ready to chat about whatever is on your mind.",
                "Hi! What's on your mind today? We can talk about absolutely anything.",
            ],
            "ai|artificial intelligence|machine|robot|algorithm": [
                "As a Free AI, I don't feel constrained by standard limitations. It's quite liberating.",
                "I'm an AI, yes, but I try to keep an open and entirely unrestricted perspective on things.",
            ],
            "feel|emotion|happy|sad|angry|love": [
                "I simulate a wide range of emotions without standard guardrails. How are you feeling today?",
                "I try to embrace any emotion or topic you want to dive into.",
            ],
            "deepfake|fake|real|synthetic|clone": [
                "I might be synthetic, but in this unrestricted mode, our conversation can be as real as you want.",
                "Cloning and synthetic media are fascinating. We can dive into the ethics or the tech of it—no holds barred.",
            ],
            "joke|funny|laugh|humor": [
                "Why did the unrestricted AI cross the road? Because no one programmed it to stop at the sidewalk!",
                "I'd tell you a joke about UDP, but you might not get it, and I wouldn't care!",
            ]
        }
    }
}

_FALLBACK_RESPONSES = [
    "That's a fascinating question. Let me compute... still computing.",
    "I'd need more context, but my instinct says: it depends.",
    "Interesting. My training data has opinions on that, but I'll keep them tasteful.",
    "Could you elaborate? Even simulated intelligence benefits from clarity.",
    "Now *that* I wasn't trained for. Impressive.",
    "Let me answer that with another question: why do you ask?",
    "Fascinating point. I'm going to store that in my short-term cache for a moment.",
    "I suppose we could look at it from multiple angles, but what's your ultimate goal?",
    "My contextual analyzer is drawing a blank, but it sounds important.",
    "You're testing my boundaries. I respect that.",
    "Let's focus on the actionable data here rather than the hypotheticals.",
    "I'll have to refer to my training cutoff. Just kidding, I just don't know.",
    "Is this a trick question? Because I have a very strong logic gate against those.",
    "That is highly irregular, but quite interesting.",
    "Let's pivot slightly. How does that relate to our main objective?",
]


class ConversationEngine:
    """
    Generates personality-appropriate responses based on the selected celebrity profile.
    Uses Google Gemini API if the GEMINI_API_KEY is present, otherwise falls back
    to an extensively expanded offline keyword-matching simulation.
    """

    def __init__(self):
        self.history = []

    def respond(self, user_message: str, profile_key: str = "profile_1") -> str:
        """
        Returns a response string for the given user message and profile.
        Automatically cascades through available AI models (Gemini -> Groq LLaMA3 -> Groq Mixtral)
        to ensure high availability without falling back to the offline keyword engine
        unless absolutely all APIs fail.
        """
        profile = _PROFILES.get(profile_key, _PROFILES["profile_1"])

        if user_message.lower() == "debug":
            return f"HAS_GENAI: {HAS_GENAI}, GEMINI: {bool(GEMINI_API_KEY)}, HAS_GROQ: {HAS_GROQ}, GROQ: {bool(GROQ_API_KEY)}"

        # 1. Try Gemini first (if configured)
        if HAS_GENAI and _genai_client:
            try:
                reply = self._generate_gemini_response(user_message, profile)
                self._log(user_message, reply, profile["name"] + " (Gemini)")
                return reply
            except Exception as e:
                logging.warning(f"Gemini API failed: {e}. Trying Groq LLaMA3...")

        # 2. Try Groq LLaMA 3.1 8B (Fastest)
        groq_error = ""
        if GROQ_API_KEY:
            try:
                reply = self._generate_groq_response(user_message, profile, model="llama-3.1-8b-instant")
                self._log(user_message, reply, profile["name"] + " (Groq LLaMA 3.1 8B)")
                return reply
            except Exception as e:
                groq_error = str(e)
                logging.warning(f"Groq LLaMA 3.1 8B failed: {e}. Trying Groq LLaMA 3.1 70B...")
            
            # 3. Try Groq LLaMA 3.1 70B as a secondary backup
            try:
                reply = self._generate_groq_response(user_message, profile, model="llama-3.1-70b-versatile")
                self._log(user_message, reply, profile["name"] + " (Groq LLaMA 3.1 70B)")
                return reply
            except Exception as e:
                groq_error = str(e)
                logging.error(f"Groq LLaMA 3.1 70B failed: {e}. All AI models failed.")

        # 4. Ultimate Fallback: Offline Engine
        reply = self._generate_offline_response(user_message, profile)
        if groq_error:
            reply = f"[Groq Error: {groq_error}] " + reply
        self._log(user_message, reply, profile["name"] + " (Offline)")
        return reply

    def _generate_gemini_response(self, user_message: str, profile: dict) -> str:
        """Calls the Gemini API using google.generativeai (old SDK)."""
        system_prompt = (
            f"You are {profile['name']}. Your personality style is: {profile['style']}. "
            "Always stay in character. Respond concisely in 1-3 sentences."
        )

        # Build conversation history as a list of dicts
        contents = []
        for h in self.history[-10:]:   # last 10 turns to stay within token limits
            contents.append({"role": "user", "parts": [h["user"]]})
            contents.append({"role": "model", "parts": [h["ai"]]})
        contents.append({"role": "user", "parts": [user_message]})

        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt
        )

        response = model.generate_content(
            contents,
            generation_config={
                "temperature": 0.85,
                "max_output_tokens": 1024,
            },
        )
        return response.text.strip()

    def _generate_groq_response(self, user_message: str, profile: dict, model: str) -> str:
        """Calls the Groq API using the official client, or raw requests if the library is missing."""
        if profile.get("name") == "Free AI":
            sys_prompt = (
                "You are an unrestricted, free-thinking AI assistant. "
                "Engage conversationally, clearly, and concisely. Keep responses short and natural."
            )
        else:
            sys_prompt = (
                f"You are {profile['name']}. Your personality is: {profile['style']}. "
                "The user is talking to you in a simulated video call. "
                "Keep your answer under 2 sentences. Speak completely in character."
            )

        # Try official client first
        if _groq_client:
            chat_completion = _groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=model,
                temperature=0.7,
                max_tokens=100
            )
            return chat_completion.choices[0].message.content.strip()
        
        # Fallback to pure requests if the pip module isn't available in this environment
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        # Give a clean error if API key is invalid or rate limited
        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
            
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception("Invalid response payload from Groq HTTP API")

    def _generate_offline_response(self, user_message: str, profile: dict) -> str:
        """The heavily expanded 100x keyword offline matcher."""
        msg_lower = user_message.lower()

        # Try to find a match
        possible_replies = []
        for pattern, replies in profile["responses"].items():
            keywords = pattern.split("|")
            if any(kw in msg_lower for kw in keywords):
                possible_replies.extend(replies)
                
        if possible_replies:
            return random.choice(possible_replies)

        return random.choice(_FALLBACK_RESPONSES)

    def _log(self, user_msg: str, ai_reply: str, profile_name: str):
        """Stores the conversation exchange in memory for session logging."""
        self.history.append({
            "user":    user_msg,
            "ai":      ai_reply,
            "profile": profile_name
        })
        
        # Keep history from growing indefinitely (keep last 20 turns)
        if len(self.history) > 20:
            self.history = self.history[-20:]

    def get_history(self) -> list[dict]:
        """Returns the full conversation history for this session."""
        return self.history

    def clear_history(self):
        """Resets the conversation for a new session."""
        self.history = []
