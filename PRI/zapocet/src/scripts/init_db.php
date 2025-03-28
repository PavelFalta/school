<?php

require_once __DIR__ . '/../vendor/autoload.php';

$dotenv = Dotenv\Dotenv::createImmutable(__DIR__ . '/..');
$dotenv->load();

try {
    $pdo = new PDO(
        "pgsql:host={$_ENV['POSTGRES_HOST']};dbname={$_ENV['POSTGRES_DB']}",
        $_ENV['POSTGRES_USER'],
        $_ENV['POSTGRES_PASSWORD']
    );
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    // Create users table
    $pdo->exec("
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ");

    // Create available_feeds table
    $pdo->exec("
        CREATE TABLE IF NOT EXISTS available_feeds (
            id SERIAL PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            url VARCHAR(512) NOT NULL,
            description TEXT,
            category VARCHAR(50)
        )
    ");

    // Create user_feeds table
    $pdo->exec("
        CREATE TABLE IF NOT EXISTS user_feeds (
            user_id INTEGER REFERENCES users(id),
            feed_id INTEGER REFERENCES available_feeds(id),
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, feed_id)
        )
    ");

    // Insert some sample feeds
    $pdo->exec("
        INSERT INTO available_feeds (title, url, description, category)
        VALUES 
            ('TechCrunch', 'https://techcrunch.com/feed/', 'Latest technology news and startup coverage', 'Technology'),
            ('BBC News', 'http://feeds.bbci.co.uk/news/rss.xml', 'Latest news from BBC', 'News'),
            ('Reuters Technology', 'https://www.reuters.com/technology/rss', 'Technology news from Reuters', 'Technology')
        ON CONFLICT DO NOTHING
    ");

    echo "Database initialized successfully!\n";
} catch (PDOException $e) {
    echo "Error: " . $e->getMessage() . "\n";
    exit(1);
} 