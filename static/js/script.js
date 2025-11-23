// --- Typewriter Effect ---
function typeWriterEffect(element, text, speed = 15) {
    let i = 0;
    const interval = setInterval(() => {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            // Auto scroll to bottom
            const chatBox = document.getElementById('chat-box');
            if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
        } else clearInterval(interval);
    }, speed);
}

// --- Auto Scroll on Load ---
window.onload = () => {
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
};

const form = document.querySelector('.input-form');
const input = document.querySelector('textarea[name="user_message"]');
const chatBox = document.getElementById('chat-box');

// --- Auto Resize Textarea ---
if (input) {
    input.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') this.style.height = '24px';
    });

    // Enter to submit, Shift+Enter for new line
    input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
    });
}

// --- Submit Event ---
if (form && input && chatBox) {
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = input.value.trim();
        if (!message) return;

        // Add User Message
        const userDiv = document.createElement('div');
        userDiv.className = 'message user';
        userDiv.innerHTML = `
            <div class="message-content">
                <div class="avatar user-avatar">
                    <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg" color="white"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                </div>
                <div class="text">${message}</div>
            </div>
        `;
        chatBox.appendChild(userDiv);

        // Add "Typing..." Message
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot';
        typingDiv.innerHTML = `
            <div class="message-content">
                <div class="avatar bot-avatar">
                    <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg" color="white"><path d="M12 2a10 10 0 1 0 10 10H12V2z"></path><path d="M12 2a10 10 0 0 1 10 10h-10V2z"></path><path d="M12 12L2.5 7.5"></path><path d="M12 12l9.5-4.5"></path></svg>
                </div>
                <div class="text"><em>Thinking...</em></div>
            </div>
        `;
        chatBox.appendChild(typingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        // Reset Input
        input.value = '';
        input.style.height = '24px';

        // AJAX Request
        fetch('/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ user_message: message })
        })
            .then(res => res.text())
            .then(html => {
                typingDiv.remove();

                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');

                // Get the last bot message from the response
                const botMsgContainer = doc.querySelector('.message.bot:last-of-type .text');

                if (botMsgContainer) {
                    const msgDiv = document.createElement('div');
                    msgDiv.className = 'message bot';
                    msgDiv.innerHTML = `
                    <div class="message-content">
                        <div class="avatar bot-avatar">
                            <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg" color="white"><path d="M12 2a10 10 0 1 0 10 10H12V2z"></path><path d="M12 2a10 10 0 0 1 10 10h-10V2z"></path><path d="M12 12L2.5 7.5"></path><path d="M12 12l9.5-4.5"></path></svg>
                        </div>
                        <div class="text"></div>
                    </div>
                `;
                    chatBox.appendChild(msgDiv);

                    // We need to extract just the text content for the typewriter effect, 
                    // OR we can just insert the HTML if we don't want the effect on complex HTML.
                    // For now, let's try to preserve the HTML structure if possible, 
                    // but the typewriter effect is simple text appending.
                    // Let's just use the innerHTML from the response directly for complex content,
                    // or use textContent for the effect.

                    // If the response has HTML (like confidence info), the typewriter effect might break it.
                    // Let's check if there's confidence info.
                    if (botMsgContainer.querySelector('.confidence-info')) {
                        msgDiv.querySelector('.text').innerHTML = botMsgContainer.innerHTML;
                    } else {
                        typeWriterEffect(msgDiv.querySelector('.text'), botMsgContainer.textContent.trim(), 10);
                    }

                    chatBox.scrollTop = chatBox.scrollHeight;
                }
            })
            .catch(err => {
                console.error(err);
                typingDiv.innerHTML = `
                <div class="message-content">
                    <div class="avatar bot-avatar">
                        <svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" height="20" width="20" xmlns="http://www.w3.org/2000/svg" color="white"><path d="M12 2a10 10 0 1 0 10 10H12V2z"></path><path d="M12 2a10 10 0 0 1 10 10h-10V2z"></path><path d="M12 12L2.5 7.5"></path><path d="M12 12l9.5-4.5"></path></svg>
                    </div>
                    <div class="text" style="color: #ff5722;">Error: Could not reach the server.</div>
                </div>
            `;
            });
    });
}
