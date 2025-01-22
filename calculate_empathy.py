from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Initialize sentiment and emotion analysis tools
sentiment_analyzer = SentimentIntensityAnalyzer()
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",device=0)

# Function to calculate empathy score
def empathy_score(response):
    # 1. Sentiment Analysis (VADER)
    sentiment = sentiment_analyzer.polarity_scores(response)
    sentiment_score = sentiment['pos'] - sentiment['neg']  # positive sentiment suggests empathy
    
    # 2. Emotion Analysis
    emotions = emotion_analyzer(response,truncation=True)
    print(emotions)
    empathy_emotion_score = 0
    for emotion in emotions:
        if emotion['label'] in ["compassion", "caring", "empathy"]:
            empathy_emotion_score += emotion['score']
            print(emotion)
    
    # 3. Key empathy phrases
    empathy_phrases = ["I understand", "I'm sorry", "That must be hard", "I can imagine"]
    phrase_score = sum([1 for phrase in empathy_phrases if phrase.lower() in response.lower()])
    
    # Final Empathy Score (combining all metrics)
    total_score = (sentiment_score * 0.4) + (empathy_emotion_score * 0.4) + (phrase_score * 0.2)
    return total_score

# Test the function
chresponse = "I'm really sorry you're feeling this way, but I'm glad you're reaching out. It's important to recognize that your feelings of worthlessness don’t reflect your true value. It can help to break down overwhelming thoughts by setting small, manageable goals and focusing on what you can do today. Consider speaking to a therapist or counselor, who can help guide you through these feelings and provide support. Sometimes our minds get clouded by negative thoughts, and professional help can provide clarity and strategies to work through them. You're not alone in this."
mhresponse1 = "It sounds like you may be struggling with depression. Depression can make you feel overwhelmed and paralyzed to change. I would suggest that you connect with a provider who can help you get to the root of where the worthlessness is coming from and help you develop a plan for recovery. In the meantime, small steps can go a long way. Self-care interventions such as journaling your feelings, mindfulness meditation, and regular exercise are all helpful to reconnecting with the present moment and gaining internal motivation. Focusing on one day at a time and bringing your thoughts back to the present can also be beneficial. There is hope!"
mhresponse2 = "Hi there, I'm sorry you're feeling this way. Let me see if I can guide you in the right direction. Often when I talk to my clients about feelings of worthlessness we start with a little bit of self-exploration. We start with noticing. Start to notice when these feelings come up for you. Is there a particular time of day, a specific person who brings it out, a phrase you hear? Just start to notice. Usually it's tied to something but it may take a while to figure out what that something/someone is. Try to be patient.Next we start to explore. When did these feelings start? Where do you think they come from? Is there something - a statement - perhaps that repeats in your head over and over again? If so, whose voice is it? These are difficult questions, and just a few of them, so take your time answering them. (We usually do it over a few sessions.) It might even be helpful to write them down somewhere. If you have a journal that would be a great place as research has shown that our brain works differently when we put pen to paper versus typing on a computer. Now comes the good news. Our brains are able to rewire themselves. This allows us to change habits we don't want as well as statements we say to ourselves that are no longer serving us. The next step is to select an ally. Someone who is or has been in your corner, someone who is always rooting for you. If you don't have someone like that, that's ok - a lot of us don't - you can just make someone up. Close your eyes and try to describe that person in great detail from the way they look to the way they act to the way they sound. Now, pick a phrase you would like that person to say to you whenever you start to think that you're worthless. Something that will help you feel better about yourself - a characteristic, a skill, a great joke you tell, a physical attribute. This also takes time and may involve you asking for help from someone who knows you.Once you have all of that together - the noticing, answers from where these feelings and statement(s) come from, your ally, your new statement, you can try to put it altogether. When the feelings come up, notice what is bringing them up and then call upon your ally to try to change the statement in your head from the self-defeating one to the more positive, uplifting one. I hope this was helpful. Again, I do this with my clients over quite a few weeks, if not months, and I am there with them the whole time. It is quite an involved process and can bring up a lot of very difficult feeling/memories. If at any point you find it too hard to go at alone, please seek help. If you take anything away from this reply, know that there is help out there and that it is possible to change the way you feel."
mhresponse3 = "First thing I'd suggest is getting the sleep you need or it will impact how you think and feel. I'd look at finding what is going well in your life and what you can be grateful for. I believe everyone has talents and wants to find their purpose in life. I think you can figure it out with some help."
mhresponse4 = "I'm sorry to hear you're feeling this intense emotion of worthlessness.Â  I'm glad to hear this has not reached the point of suicidal ideation; however, it does sounds like you could use some additional support right now.Â  I would recommend seeking out counseling to help you challenge the negative beliefs you have about yourself.Â  Although many types of therapy would be helpful, cognitive-behavioral therapy has been shown to be a good approach for this type of struggle.Â  A CBT therapist can help you identify your negative thoughts and beliefs, figure out the ways your thoughts are being distorted (for example, all-or-nothing thinking, or discounting the positives about yourself), and reframe your thoughts to be more positive.Â  You might also consider EMDR therapy, which helps the brain reprocess traumatic or distressing memories and helps you move forward with more positive beliefs about yourself.Â  Best wishes!"
score1 = empathy_score(chresponse)
score2 = empathy_score(mhresponse1)
score3 = empathy_score(mhresponse2)
score4 = empathy_score(mhresponse3)
score5 = empathy_score(mhresponse4)
print("Empathy Score - chatgpt: ", score1)
print("Empathy Score - mhresponse1: ", score2)
print("Empathy Score - mhresponse2: ", score3)
print("Empathy Score - mhresponse3: ", score4)
print("Empathy Score - mhresponse4: ", score5)
