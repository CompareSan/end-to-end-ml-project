variable "cluster_name" {
    type = string
    default = "terraformk8s"
}

variable "k8s_version" {
    type = string
    default = "1.29.1-do.0"
}

variable "region" {
    type = string
    default = "fra1"
}
