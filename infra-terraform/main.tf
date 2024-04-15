terraform {
  required_providers {
    digitalocean = {
        source = "digitalocean/digitalocean"
    }
  }
}

terraform {
  backend "s3" {
    endpoint                    = "https://fra1.digitaloceanspaces.com"
    key                         = "terraform.tfstate"
    bucket                      = "terraform-state-files-ml-project"
    region                      = "eu-central-1"
    skip_requesting_account_id  = true
    skip_credentials_validation = true
    skip_metadata_api_check     = true
    skip_s3_checksum            = true
  }
}


provider "digitalocean" {
  token = var.do_token
}


resource "digitalocean_kubernetes_cluster" "terraformk8s" {
  name = var.cluster_name
  region = var.region
  version = var.k8s_version
  node_pool {
    name = "worker-node-pool"
    size = "s-2vcpu-4gb"
    auto_scale = true
    max_nodes = 3
    min_nodes = 2
  }
}
