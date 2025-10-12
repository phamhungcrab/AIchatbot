// --- Hiệu ứng gõ từng ký tự ---
function typeWriterEffect(element, text, speed = 15) {
    let i = 0;
    const interval = setInterval(() => {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            element.scrollTop = element.scrollHeight;
        } else clearInterval(interval);
    }, speed);
}

// --- Cuộn cuối khi tải ---
window.onload = () => {
    const chatBox = document.getElementById('chat-box');
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
};

const form = document.querySelector('.input-form');
const input = document.querySelector('input[name="user_message"]');
const chatBox = document.getElementById('chat-box');

// --- Sự kiện gửi tin nhắn ---
if (form && input && chatBox) {
    form.addEventListener('submit', (e) => {
        e.preventDefault(); // Ngăn reload
        const message = input.value.trim();
        if (!message) return;

        // Thêm tin nhắn người dùng
        const userDiv = document.createElement('div');
        userDiv.className = 'message user';
        userDiv.textContent = message;
        chatBox.appendChild(userDiv);

        // “Bot đang gõ...”
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot';
        typingDiv.innerHTML = '<em>Bot đang gõ...</em>';
        chatBox.appendChild(typingDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        // Gửi request AJAX
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

        // Lấy tin nhắn bot cuối cùng
        const botMsg = doc.querySelector('.message.bot:last-of-type');
        if (botMsg) {
            const msg = document.createElement('div');
            msg.className = 'message bot';
            chatBox.appendChild(msg);
            typeWriterEffect(msg, botMsg.textContent.trim(), 20);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        });


        input.value = '';
    });
}

// --- Enter để gửi ---
if (input) {
    input.addEventListener('keypress', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
    });
}
