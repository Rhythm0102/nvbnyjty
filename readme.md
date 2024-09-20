# Context Aware Chatbot

This project solves problem statement 4 i.e. context-aware chatbot that addresses three key challenges:

1. **Knowledge integration**: The chatbot responds strictly from provided documents (PDFs, text files, etc.), avoiding any pre-trained information.
2. **Task Execution**: The chatbot can perform tasks like booking a policy or fetching an invoice using API access (for demonstration purposes we have emulated API which can then be replaced with real ones) .
3. **Context Maintenance**: It maintains conversation context across multiple interactions, remembering previous discussions with the user.

## Solution Overview

The solution is divided into three components, each addressing one of the problems mentioned, Overall, we are using Rag approach for this but enchantment:

### 1. Document-based Response System
- **Input**: A collection of PDFs and other documents.
- **Process**:
  - Extracts text and images from the documents.
  - Text is passed through an embedding model and stored in a vector space (vector database).
- **Query Handling**:
  - When a user query is received, first the query is rewritten by another smaller LLM to make it better for vector store retrieval and to fix any spelling or grammar error in, future it can also be used to translate query t9o serve people of diverse backgrounds 
 - Then the query is converted into a vector embedding.
  - The chatbot retrieves relevant information from the vector database using a combination of BM25 and vector similarity search, followed by a re-ranking of the most relevant chunks using a cross-encoder.
  - The response is generated using this retrieved information while also taking into account the past conversation.

### 2. Action Execution Module
- The chatbot is not only capable of providing information but also executing a variety of tasks by integrating with external APIs.
- **API Integration**: The chatbot can perform actions like creating or canceling orders, collecting payments via a link, viewing invoices, and more, based on provided APIs.
- **Process**:
  - The assistant identifies the required function to call based on user input.
  - Calls the appropriate API with the correct parameters.
  - The response from the API is returned to the user in a human-readable format.
- **Example Actions**:
  - Creating/Cancelling orders.
  - Collecting payments via a payment link.
  - Viewing and retrieving invoices.
 - To book any policy just say that to the bot along with your name, email, and number it will return an       order ID to show the booking compilation, or ask to give an invoice for the order id it will return an link as a demonstration  ( this is for demonstration only just to show our bot is capable of action based on query, for real-world we will  integrate real APIs for task execution)
  
- **Subtasks**:
  - Implemented an API integration layer that connects the chatbot to various external services.
  - Ensured API responses are formatted and presented clearly to the user.

# Running the chatbot application

## Prerequisites:

- **Python:** Ensure you have Python installed on your system. You can check by running `python --version` or `python3 --version` in your terminal. If not installed, download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).
- **Text Editor/IDE:** Choose a text editor or Integrated Development Environment (IDE) for writing your Python code. Popular options include Visual Studio Code, PyCharm, Sublime Text, and others.

## Steps:

1. **Navigate to the Folder:**
   - Open a file explorer or terminal window.
   - Locate the directory containing your Python application files ( where `main.py` and `requirements.txt` reside).
   - Use the `cd` command in the terminal to navigate to this directory. For example, if your application is in a folder named `my_app` on your Desktop, type:

     ```bash
     cd Desktop/my_app
     ```

2. **Open Terminal:**
   - Once you've located your application directory, launch a terminal window within that folder. 

3. **Install Dependencies: **
   - Run the following command to install the dependencies using pip:

     ```bash
     pip install -r requirements.txt
     ```

     This command reads `requirements.txt` and installs the necessary libraries.

4. **Run the Application:**
   - Once dependencies are installed, execute the application script using the following command:

     ```bash
     python app.py
     ```

  

   - When successful, the application will start acting as a local server 

**5. Accessing the Application on a local server:**
  - use curl commands as described in the next step to send request to the local server.

**6. Using `curl` Commands :**

- How to Upload Documents

To upload documents for the chatbot to process, use the following curl command:

```bash
curl -X POST -H "Content-Type: application/json" -d "{\"folder_path\": \"yourfilepath"}" http://localhost:3000/upload
```
Edit the 'yourfilepath' section as per the file location

- How to Ask Queries

To ask queries related to it, use the following curl command:

```bash
curl --location "http://localhost:3000/chat" --header "Content-Type: application/json" --header "x-user-id: 123" --header "x-session-id: J5K7P1ZQ" --data "{\"query\": \"yourquery"}"
```

Edit the 'your query' section as per your query.

**7. current limitation:**
As it has a prototype it still has some limitations 
-	The actions assistant takes like booking and invoice link generation are done using emulated API I.e. those are faked just for demonstration, although for the assistant it’s like receiving real API responses 
-	The API endpoint given in the application does not accept new documents as storage and updating was not supported on  basic tier(on server) it will answer based on the file in uploads folder(which is multiple real world policy documents in one ) from the GitHub repo , although the code does support updating of knowledge base to explore this we request you to run it locally to explore full functionality 
-	After calling upload command for first time it maybe take some time to process the documents as it  needs to create embeddings for the text although this is only done once 


## Video Demo

[Click here to watch the video](https://youtu.be/MiXiGvHg_QE)






