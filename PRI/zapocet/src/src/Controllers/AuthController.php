<?php

namespace App\Controllers;

use App\Models\User;
use Firebase\JWT\JWT;
use Firebase\JWT\Key;

class AuthController {
    private $user;

    public function __construct() {
        $this->user = new User();
    }

    public function register(): void {
        $data = json_decode(file_get_contents('php://input'), true);
        
        if (!isset($data['username']) || !isset($data['email']) || !isset($data['password'])) {
            http_response_code(400);
            echo json_encode(['error' => 'Missing required fields']);
            return;
        }

        $existingUser = $this->user->findByEmail($data['email']);
        if ($existingUser) {
            http_response_code(400);
            echo json_encode(['error' => 'Email already registered']);
            return;
        }

        $userId = $this->user->create(
            $data['username'],
            $data['email'],
            $data['password']
        );

        if ($userId) {
            http_response_code(201);
            echo json_encode(['message' => 'User registered successfully']);
        } else {
            http_response_code(500);
            echo json_encode(['error' => 'Registration failed']);
        }
    }

    public function login(): void {
        $data = json_decode(file_get_contents('php://input'), true);
        
        if (!isset($data['email']) || !isset($data['password'])) {
            http_response_code(400);
            echo json_encode(['error' => 'Missing required fields']);
            return;
        }

        $user = $this->user->findByEmail($data['email']);
        if (!$user || !$this->user->verifyPassword($data['password'], $user['password_hash'])) {
            http_response_code(401);
            echo json_encode(['error' => 'Invalid credentials']);
            return;
        }

        $token = $this->generateToken($user['id']);
        
        http_response_code(200);
        echo json_encode([
            'token' => $token,
            'user' => [
                'id' => $user['id'],
                'username' => $user['username'],
                'email' => $user['email']
            ]
        ]);
    }

    public function getCurrentUser(): void {
        $userId = $this->getUserIdFromToken();
        if (!$userId) {
            http_response_code(401);
            echo json_encode(['error' => 'Unauthorized']);
            return;
        }

        $user = $this->user->findById($userId);
        if (!$user) {
            http_response_code(404);
            echo json_encode(['error' => 'User not found']);
            return;
        }

        http_response_code(200);
        echo json_encode($user);
    }

    private function generateToken(int $userId): string {
        $payload = [
            'user_id' => $userId,
            'exp' => time() + $_ENV['JWT_EXPIRATION']
        ];

        return JWT::encode($payload, $_ENV['JWT_SECRET'], 'HS256');
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