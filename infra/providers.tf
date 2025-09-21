terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = { source = "hashicorp/azurerm", version = "~> 4.0" }
    azuread = { source = "hashicorp/azuread", version = "~> 2.50" }
    random  = { source = "hashicorp/random", version = "~> 3.6" }
    local   = { source = "hashicorp/local", version = "~> 2.5" }
    null    = { source = "hashicorp/null", version = "~> 3.2" }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

provider "azuread" {}
