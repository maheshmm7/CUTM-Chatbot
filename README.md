# CUTM CHATBOT - An AI-Powered University Assistant

A smart chatbot implementation designed for Centurion University of Technology and Management (CUTM). This project serves as an intelligent virtual assistant to help students, faculty, and visitors with university-related queries. Built using PyTorch and deployed with Flask and JavaScript, this chatbot demonstrates the practical application of AI in enhancing university communication systems.

## About CUTM CHATBOT

CUTM CHATBOT is a college project developed for Centurion University of Technology and Management. It's designed to:
- Assist with common university-related queries
- Provide information about courses and programs
- Help with administrative questions
- Guide visitors through university resources
- Support students with general inquiries

## Features

- Custom neural network implementation using PyTorch
- Natural Language Processing with NLTK
- Multiple deployment options:
  - Integrated Flask application with Jinja2 templates
  - Standalone REST API with separate frontend
- Real-time chat interface with JavaScript
- Customizable intents and responses specific to CUTM
- Easy-to-modify architecture

## Deployment Options

### 1. Full Stack Flask Application
- Complete integration with Flask
- Server-side rendering using Jinja2 templates
- All-in-one solution

### 2. Decoupled Architecture
- Separate frontend and backend
- Flask serves only as REST API
- Frontend can be integrated into any application
- Greater flexibility and scalability

## Prerequisites

- Python 3.7+
- Basic knowledge of Python, Flask, and JavaScript
- Understanding of REST APIs
- Familiarity with PyTorch (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/python-engineer/chatbot-deployment.git
cd chatbot-deployment
```

2. Create and activate virtual environment:
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install Flask torch torchvision nltk
```

4. Install NLTK data:
```python
python
>>> import nltk
>>> nltk.download('punkt')
```

## Configuration

1. The chatbot comes pre-configured with CUTM-specific intents in `intents.json`:
```json
{
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey"],
            "responses": ["Hello! Welcome to CUTM Chatbot!", "Hi there! How can I help you with CUTM related queries?", "Hey! Ask me anything about Centurion University!"]
        },
        {
            "tag": "courses",
            "patterns": ["What courses are offered?", "Tell me about programs", "Available degrees"],
            "responses": ["CUTM offers various undergraduate and postgraduate programs in Engineering, Management, Pharmacy, and more. Would you like specific information about any program?"]
        }
        // Add more CUTM-specific intents here
    ]
}
```

2. Train the model:
```bash
python train.py
```

3. Verify the training by testing in console:
```bash
python chat.py
```

## Running the Application

### Option 1: Integrated Flask App
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

### Option 2: API-Only Mode
```bash
python app.py --api-only
```
The API will be available at `http://localhost:5000/predict`.

## API Documentation

### Prediction Endpoint
- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
    "message": "Tell me about CUTM"
}
```
- **Response**:
```json
{
    "answer": "Centurion University of Technology and Management is a premier educational institution focused on skill-integrated education."
}
```

## Customization

### Adding New Intents
1. Add new patterns and responses to `intents.json`
2. Retrain the model using `train.py`
3. Restart the Flask application

### Modifying the Neural Network
Edit `model.py` to customize the neural network architecture:
- Adjust layer sizes
- Modify activation functions
- Add or remove layers

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make changes
4. Commit (`git commit -am 'Add new feature'`)
5. Push (`git push origin feature/improvement`)
6. Create a Pull Request

## Troubleshooting

Common issues and solutions:

1. **Model not loading**:
   - Ensure `data.pth` exists
   - Verify PyTorch version compatibility

2. **NLTK errors**:
   - Run `nltk.download('punkt')` in Python console
   - Check NLTK installation

3. **Flask application not starting**:
   - Verify virtual environment is activated
   - Check port 5000 is available

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Centurion University of Technology and Management
- Faculty mentors and guides
- Original PyTorch chatbot tutorial
- Flask documentation
- NLTK community
- PyTorch team

## Contact

For questions and support about CUTM CHATBOT, please open an issue in the GitHub repository or contact the development team at Centurion University of Technology and Management.