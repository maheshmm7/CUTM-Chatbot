class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        };

        this.state = false;
        this.messages = [];
        this.predefinedPrompts = [
            "Hello! How can I assist you today?",
            "Admission criteria",
            "Where is CUTM located?",
            "What facilities are available?",
            "What courses are available?"
        ];
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;

        if (this.state) {
            chatbox.classList.add('chatbox--active');
            if (this.messages.length === 0) {
                this.displayInitialPrompts(chatbox);
            }
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    displayInitialPrompts(chatbox) {
        this.predefinedPrompts.forEach(prompt => {
            this.messages.push({ name: "CUTM BOT", message: prompt });
        });
        this.updateChatText(chatbox);
    }

    onSendButton(chatbox) {
        const textField = chatbox.querySelector('input');
        const text1 = textField.value.trim();

        if (text1 === "") {
            return;
        }

        // Clear the input field after the message is sent
        textField.value = '';

        const msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);
        this.updateChatText(chatbox);

        fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            const msg2 = { name: "CUTM BOT", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
        })
        .catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox);
        });
    }

    updateChatText(chatbox) {
        let html = '';
        this.messages.forEach(item => {
            if (item.name === "User") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
        chatmessage.scrollTop = chatmessage.scrollHeight;
    }
}

const chatbox = new Chatbox();
chatbox.display();
