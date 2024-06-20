css = '''
<style>
.chat-message {
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
    display: flex;
}

.chat-message.user {
    background-color: #2E5390; /* ChatGPT-like color */
}

.chat-message.bot {
    background-color: #062E70; /* ChatGPT-like color */
}

.chat-message .avatar {
    width: 15%;
}

.chat-message .avatar img {
    max-width: 40px;
    max-height: 40px; /* Corrected typo */
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 60%;
    padding-left: 0.1rem; /* Corrected typo */
    color: #fff;
}
'''

bot_template = '''
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://i.imgur.com/6vp0O6Db.jpg" alt="User picture" >
        </div>
        <div class="message" style="font-size: 20px;">{{MSG}}</div>
    </div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.imgur.com/HYcn9xO.png" alt="User picture" >
    </div>
    <div class="message" style="font-size: 20px;">{{MSG}}</div>
</div>

'''