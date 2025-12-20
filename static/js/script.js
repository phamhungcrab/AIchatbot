// --- AI Chatbot Script ---
// Enter key submit + Auto scroll + Auto resize

document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('.input-form');
    const input = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');

    // --- Auto Scroll on Load ---
    if (chatBox) {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // --- Auto Resize Textarea ---
    if (input) {
        input.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
            if (this.value === '') this.style.height = '24px';
        });

        // 🎯 Enter to submit, Shift+Enter for new line
        input.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
                e.preventDefault();
                if (this.value.trim()) {
                    form.submit();
                }
            }
        });

        // Focus on input on load
        input.focus();
    }
});
