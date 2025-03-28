<?php

namespace App\Models;

use App\Config\Database;
use PDO;

class Feed {
    private $db;

    public function __construct() {
        $this->db = Database::getInstance()->getConnection();
    }

    public function getAll(): array {
        $sql = "SELECT * FROM available_feeds ORDER BY title";
        $stmt = $this->db->query($sql);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }

    public function getUserFeeds(int $userId): array {
        $sql = "SELECT f.* FROM available_feeds f
                JOIN user_feeds uf ON f.id = uf.feed_id
                WHERE uf.user_id = :user_id
                ORDER BY f.title";
        
        $stmt = $this->db->prepare($sql);
        $stmt->execute([':user_id' => $userId]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }

    public function addUserFeed(int $userId, int $feedId): bool {
        try {
            $sql = "INSERT INTO user_feeds (user_id, feed_id) VALUES (:user_id, :feed_id)";
            $stmt = $this->db->prepare($sql);
            return $stmt->execute([
                ':user_id' => $userId,
                ':feed_id' => $feedId
            ]);
        } catch (PDOException $e) {
            // Ignore duplicate key errors
            if ($e->getCode() == 23505) { // Unique violation
                return true;
            }
            return false;
        }
    }

    public function removeUserFeed(int $userId, int $feedId): bool {
        $sql = "DELETE FROM user_feeds WHERE user_id = :user_id AND feed_id = :feed_id";
        $stmt = $this->db->prepare($sql);
        return $stmt->execute([
            ':user_id' => $userId,
            ':feed_id' => $feedId
        ]);
    }

    public function getFeedContent(string $url): ?string {
        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
        curl_setopt($ch, CURLOPT_USERAGENT, 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36');
        
        $content = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($httpCode === 200) {
            return $content;
        }
        return null;
    }

    public function parseFeed(string $content): ?array {
        libxml_use_internal_errors(true);
        $xml = simplexml_load_string($content);
        
        if ($xml === false) {
            return null;
        }

        $items = [];
        foreach ($xml->channel->item as $item) {
            $items[] = [
                'title' => (string)$item->title,
                'link' => (string)$item->link,
                'description' => (string)$item->description,
                'pubDate' => (string)$item->pubDate,
                'guid' => (string)$item->guid
            ];
        }

        return $items;
    }
} 