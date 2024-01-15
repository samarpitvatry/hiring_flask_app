
# Hiring Visualization Flask App

## Description
This Flask application provides visual analytics related to hiring processes. It includes various visualizations like hiring duration analysis, distribution by career level, and more, leveraging data processing and visualization libraries.

## Project Structure
- `app.py`: Main Flask application file with backend logic.
- `Dockerfile`: Contains instructions to containerize the application.
- `requirements.txt`: Lists dependencies required for the application.
- `static/`: Directory for static content, primarily images.
- `templates/`: Contains HTML templates for rendering web pages.

## Setup and Running the Application

### Prerequisites
- Python 3.8
- Docker (optional for containerized deployment)

### Installation Steps
1. Clone the repository or download the source code.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application
- To run the app locally:
  ```
  python app.py
  ```
- To run the app using Docker:
  1. Build the Docker image:
     ```
     docker build -t hiring_flask_app .
     ```
  2. Run the Docker container:
     ```
     docker run -p 5004:5004 hiring_flask_app
     ```

## Dependencies
- Flask
- Werkzeug
- pandas
- matplotlib
- seaborn
- numpy
- Pillow
- openpyxl
- plotly
