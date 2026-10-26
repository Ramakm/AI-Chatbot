terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

# For WSL2 and Linux, Docker daemon communicates via the Unix socket: unix:///var/run/docker.sock
# For native Windows PowerShell, override with: npipe:////./pipe/docker_engine
variable "docker_host" {
  type        = string
  description = "Docker daemon socket or named pipe"
  default     = "unix:///var/run/docker.sock"
}

variable "nginx_port" {
  type        = number
  description = "External host port for Nginx reverse proxy"
  default     = 8081
}

provider "docker" {
  host = var.docker_host
}

# Build the chatbot image
resource "docker_image" "chatbot" {
  name = "ai-chatbot:latest"

  build {
    context    = abspath(path.module)
    dockerfile = "Dockerfile"
    tag        = ["ai-chatbot:latest"]
  }

  keep_locally = true
}

# Download the Nginx image
resource "docker_image" "nginx" {
  name         = "nginx:alpine"
  keep_locally = true
}

# Create a Docker network
resource "docker_network" "chatbot_network" {
  name = "chatbot-network"
}

# Create the chatbot container
resource "docker_container" "chatbot" {
  name  = "ai-chatbot"
  image = docker_image.chatbot.image_id

  restart = "unless-stopped"

  env = [
    "WORKERS=4",
    "LOG_LEVEL=info"
  ]

  ports {
    internal = 8000
    external = 8000
  }

  mounts {
    source = abspath("${path.module}/templates")
    target = "/app/templates"
    type   = "bind"
  }

  mounts {
    source = abspath("${path.module}/static")
    target = "/app/static"
    type   = "bind"
  }

  mounts {
    source = abspath("${path.module}/intents.json")
    target = "/app/intents.json"
    type   = "bind"
  }

  networks_advanced {
    name    = docker_network.chatbot_network.name
    aliases = ["chatbot"]
  }

  healthcheck {
    test         = ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval     = "30s"
    timeout      = "10s"
    retries      = 3
    start_period = "40s"
  }
}

# Create the Nginx reverse-proxy container
resource "docker_container" "nginx" {
  name  = "chatbot-proxy"
  image = docker_image.nginx.image_id

  restart = "unless-stopped"

  ports {
    internal = 80
    external = var.nginx_port
  }

  mounts {
    source    = abspath("${path.module}/nginx.conf")
    target    = "/etc/nginx/nginx.conf"
    type      = "bind"
    read_only = true
  }

  networks_advanced {
    name = docker_network.chatbot_network.name
  }

  depends_on = [docker_container.chatbot]
}

output "chatbot_direct_url" {
  value       = "http://localhost:8000"
  description = "Direct URL to FastAPI Chatbot"
}

output "nginx_proxy_url" {
  value       = "http://localhost:${var.nginx_port}"
  description = "URL through Nginx reverse proxy"
}

