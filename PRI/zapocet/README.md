# RSS Feed Aggregator

A web application that allows users to create accounts, select their preferred RSS feeds, and view them in a clean, organized interface.

## Requirements

- Docker
- Docker Compose

## Setup

1. Clone the repository
2. Create a `.env` file in the root directory (copy from `.env.example`)
3. Run the following commands:

```bash
# Build and start the containers
docker-compose up -d --build

# Install PHP dependencies
docker-compose exec php composer install

# Create database tables
docker-compose exec php php src/scripts/init_db.php
```

## Access the Application

The application will be available at http://localhost:8080

## Development

- Frontend code is in `src/public`
- Backend code is in `src/src`
- Database migrations are in `src/scripts`

## Features

- User registration and authentication
- RSS feed selection and management
- Feed display with article previews
- Responsive design 