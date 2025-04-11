# SimpleLangChainApp

### Create a virtual environment
```commandline
virtualenv -p ~/.pyenv/versions/3.11.6/bin/python .venv
source .venv/bin/activate
which python3
```
[See here for more detail](https://github.com/binodrajpandey/django_crud/wiki)

Or create it from the IDE

### Create OPEN_API_KEY
Create key from here https://platform.openai.com/api-keys

### crate TAVILY_API_KEY 
Create an account from [here](https://app.tavily.com/home)
copy the api key and set it in to the .env file

### Langfuse
In order for langfuse to work, configure following after running langfuse in localhost using or in remote server
LANGFUSE_SECRET_KEY=<>
LANGFUSE_PUBLIC_KEY=<>
LANGFUSE_HOST=<>