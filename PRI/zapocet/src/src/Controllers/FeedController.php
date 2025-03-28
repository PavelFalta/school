<?php

namespace App\Controllers;

use App\Models\Feed;
use App\Models\User;

class FeedController {
    private $feed;
    private $user;

    public function __construct() {
        $this->feed = new Feed();
        $this->user = new User();
    }

    public function getAllFeeds(): void {
        $feeds = $this->feed->getAll();
        http_response_code(200);
        echo json_encode($feeds);
    }

    public function getUserFeeds(): void {
        $userId = $this->getUserIdFromToken();
        if (!$userId) {
            http_response_code(401);
            echo json_encode(['error' => 'Unauthorized']);
            return;
        }

        $feeds = $this->feed->getUserFeeds($userId);
        http_response_code(200);
        echo json_encode($feeds);
    }

    public function addUserFeed(): void {
        $userId = $this->getUserIdFromToken();
        if (!$userId) {
            http_response_code(401);
            echo json_encode(['error' => 'Unauthorized']);
            return;
        }

        $data = json_decode(file_get_contents('php://input'), true);
        if (!isset($data['feed_id'])) {
            http_response_code(400);
            echo json_encode(['error' => 'Missing feed_id']);
            return;
        }

        if ($this->feed->addUserFeed($userId, $data['feed_id'])) {
            http_response_code(200);
            echo json_encode(['message' => 'Feed added successfully']);
        } else {
            http_response_code(500);
            echo json_encode(['error' => 'Failed to add feed']);
        }
    }

    public function removeUserFeed(): void {
        $userId = $this->getUserIdFromToken();
        if (!$userId) {
            http_response_code(401);
            echo json_encode(['error' => 'Unauthorized']);
            return;
        }

        $data = json_decode(file_get_contents('php://input'), true);
        if (!isset($data['feed_id'])) {
            http_response_code(400);
            echo json_encode(['error' => 'Missing feed_id']);
            return;
        }

        if ($this->feed->removeUserFeed($userId, $data['feed_id'])) {
            http_response_code(200);
            echo json_encode(['message' => 'Feed removed successfully']);
        } else {
            http_response_code(500);
            echo json_encode(['error' => 'Failed to remove feed']);
        }
    }

    public function getFeedContent(): void {
        $data = json_decode(file_get_contents('php://input'), true);
        if (!isset($data['url'])) {
            http_response_code(400);
            echo json_encode(['error' => 'Missing feed URL']);
            return;
        }

        $content = $this->feed->getFeedContent($data['url']);
        if (!$content) {
            http_response_code(404);
            echo json_encode(['error' => 'Feed not found or inaccessible']);
            return;
        }

        $items = $this->feed->parseFeed($content);
        if (!$items) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid feed format']);
            return;
        }

        http_response_code(200);
        echo json_encode($items);
    }

    private function getUserIdFromToken(): ?int {
        $headers = getallheaders();
        if (!isset($headers['Authorization'])) {
            return null;
        }

        $token = str_replace('Bearer ', '', $headers['Authorization']);
        try {
            $decoded = JWT::decode($token, new Key($_ENV['JWT_SECRET'], 'HS256'));
            return $decoded->user_id;
        } catch (\Exception $e) {
            return null;
        }
    }
} 