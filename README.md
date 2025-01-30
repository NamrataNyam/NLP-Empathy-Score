## Empathy Score Calculation  

The **Empathy Score** is a composite metric designed to assess the emotional intelligence of a conversational AI model. It evaluates the response’s sentiment, emotional depth, and engagement, ensuring the AI generates responses that are more human-like and empathetic.  

### Components of the Empathy Score  

1. **Sentiment Analysis**  
   - Uses **VADER (Valence Aware Dictionary and sEntiment Reasoner)** to analyze the sentiment of the response.  
   - Assigns a sentiment polarity score ranging from **-1 (negative)** to **+1 (positive)** based on the intensity and context of words.  

2. **Emotion Analysis**  
   - Leverages a **transformer-based model (DistilRoBERTa)** to classify emotions in the response.  
   - Assigns probability scores to emotion labels such as **joy, sadness, anger, fear, and surprise** to determine the dominant emotion.  

3. **Engagement Score**  
   - Measures how engaging and empathetic the response is by detecting words and phrases commonly used in emotionally intelligent conversations.  
   - Helps ensure that the AI's responses feel **genuine and interactive** rather than robotic.  

### Weighted Final Score Calculation  

The **final empathy score** is a weighted combination of the three components:  

Empathy Score = (0.4 × Sentiment Score) + (0.4 × Emotion Score) + (0.2 × Engagement Score)

### Applications  
- Enhancing **LLM-based conversational AI** to generate more emotionally intelligent responses.  
- Improving **chatbots for mental health support, customer service, and social interactions**.  
- Benchmarking AI empathy against human-like conversational patterns.  
  
