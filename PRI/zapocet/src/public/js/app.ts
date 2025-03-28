interface User {
    id: number;
    username: string;
    email: string;
}

interface Feed {
    id: number;
    title: string;
    url: string;
    description: string;
    category: string;
}

class App {
    private currentUser: User | null = null;
    private token: string | null = null;

    constructor() {
        this.initializeEventListeners();
        this.checkAuthStatus();
    }

    private initializeEventListeners(): void {
        // Navigation
        document.getElementById('home-link')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showHome();
        });

        document.getElementById('login-link')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showLoginForm();
        });

        document.getElementById('register-link')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showRegisterForm();
        });

        document.getElementById('logout-link')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.logout();
        });

        // Forms
        document.querySelector('#login-form form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleLogin();
        });

        document.querySelector('#register-form form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleRegister();
        });
    }

    private async checkAuthStatus(): Promise<void> {
        const token = localStorage.getItem('token');
        if (token) {
            this.token = token;
            try {
                const response = await fetch('/api/user', {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });
                if (response.ok) {
                    this.currentUser = await response.json();
                    this.updateUIForLoggedInUser();
                } else {
                    this.logout();
                }
            } catch (error) {
                console.error('Error checking auth status:', error);
                this.logout();
            }
        }
    }

    private updateUIForLoggedInUser(): void {
        document.getElementById('login-link')?.classList.add('hidden');
        document.getElementById('register-link')?.classList.add('hidden');
        document.getElementById('feeds-link')?.classList.remove('hidden');
        document.getElementById('logout-link')?.classList.remove('hidden');
        this.showAvailableFeeds();
    }

    private updateUIForLoggedOutUser(): void {
        document.getElementById('login-link')?.classList.remove('hidden');
        document.getElementById('register-link')?.classList.remove('hidden');
        document.getElementById('feeds-link')?.classList.add('hidden');
        document.getElementById('logout-link')?.classList.add('hidden');
        this.showLoginForm();
    }

    private async handleLogin(): Promise<void> {
        const email = (document.getElementById('login-email') as HTMLInputElement).value;
        const password = (document.getElementById('login-password') as HTMLInputElement).value;

        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email, password })
            });

            if (response.ok) {
                const data = await response.json();
                this.token = data.token;
                this.currentUser = data.user;
                localStorage.setItem('token', data.token);
                this.updateUIForLoggedInUser();
                this.showAvailableFeeds();
            } else {
                this.showError('Login failed. Please check your credentials.');
            }
        } catch (error) {
            console.error('Login error:', error);
            this.showError('An error occurred during login.');
        }
    }

    private async handleRegister(): Promise<void> {
        const username = (document.getElementById('register-username') as HTMLInputElement).value;
        const email = (document.getElementById('register-email') as HTMLInputElement).value;
        const password = (document.getElementById('register-password') as HTMLInputElement).value;

        try {
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, email, password })
            });

            if (response.ok) {
                this.showSuccess('Registration successful! Please login.');
                this.showLoginForm();
            } else {
                this.showError('Registration failed. Please try again.');
            }
        } catch (error) {
            console.error('Registration error:', error);
            this.showError('An error occurred during registration.');
        }
    }

    private async showAvailableFeeds(): Promise<void> {
        try {
            const response = await fetch('/api/feeds');
            if (response.ok) {
                const feeds: Feed[] = await response.json();
                this.renderFeeds(feeds);
            }
        } catch (error) {
            console.error('Error fetching feeds:', error);
        }
    }

    private renderFeeds(feeds: Feed[]): void {
        const feedsList = document.getElementById('feeds-list');
        if (!feedsList) return;

        feedsList.innerHTML = feeds.map(feed => `
            <div class="feed-card">
                <h3>${feed.title}</h3>
                <p>${feed.description}</p>
                <p><strong>Category:</strong> ${feed.category}</p>
                <button onclick="app.toggleFeed(${feed.id})">Add to My Feeds</button>
            </div>
        `).join('');
    }

    private showLoginForm(): void {
        document.getElementById('login-form')?.classList.remove('hidden');
        document.getElementById('register-form')?.classList.add('hidden');
        document.getElementById('feeds-container')?.classList.add('hidden');
        document.getElementById('my-feeds-container')?.classList.add('hidden');
    }

    private showRegisterForm(): void {
        document.getElementById('login-form')?.classList.add('hidden');
        document.getElementById('register-form')?.classList.remove('hidden');
        document.getElementById('feeds-container')?.classList.add('hidden');
        document.getElementById('my-feeds-container')?.classList.add('hidden');
    }

    private showHome(): void {
        if (this.currentUser) {
            this.showAvailableFeeds();
            document.getElementById('feeds-container')?.classList.remove('hidden');
        } else {
            this.showLoginForm();
        }
    }

    private logout(): void {
        this.currentUser = null;
        this.token = null;
        localStorage.removeItem('token');
        this.updateUIForLoggedOutUser();
    }

    private showError(message: string): void {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        document.querySelector('main')?.prepend(errorDiv);
        setTimeout(() => errorDiv.remove(), 3000);
    }

    private showSuccess(message: string): void {
        const successDiv = document.createElement('div');
        successDiv.className = 'success';
        successDiv.textContent = message;
        document.querySelector('main')?.prepend(successDiv);
        setTimeout(() => successDiv.remove(), 3000);
    }
}

// Initialize the app
const app = new App(); 