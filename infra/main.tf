locals {
  # Códigos curtos por região para nomes
  loc_codes = {
    brazilsouth = "brs"
    eastus      = "eus"
    eastus2     = "eus2"
    westeurope  = "weu"
    northeurope = "neu"
    centralus   = "cus"
    westus      = "wus"
  }

  loc_code = lookup(local.loc_codes, var.location, var.location)
  prefix   = "${var.project_name}-${var.environment}-${local.loc_code}"

  rg_name     = "rg-${local.prefix}"
  asp_name    = "asp-${local.prefix}"
  webapp_name = "webapp-${local.prefix}"
  bot_name    = "bot-${local.prefix}"
}

# Para obter o tenant_id atual
data "azurerm_client_config" "current" {}

# ---------- Resource Group ----------
resource "azurerm_resource_group" "rg" {
  name     = local.rg_name
  location = var.location
  tags     = var.tags
}

# ---------- App Service Plan (Linux) ----------
resource "azurerm_service_plan" "asp" {
  name                = local.asp_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location

  os_type  = "Linux"
  sku_name = var.sku_name

  tags = var.tags
}

# ---------- Linux Web App (Publish: code, Python 3.13) ----------
resource "azurerm_linux_web_app" "webapp" {
  name                = local.webapp_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  service_plan_id     = azurerm_service_plan.asp.id

  site_config {
    always_on        = false
    app_command_line = "python3 -m aiohttp.web -H 0.0.0.0 -P 8000 app:init_func"

    application_stack {
      python_version = "3.13"
    }
  }

  logs {
    http_logs {
      file_system {
        retention_in_days = 3
        retention_in_mb   = 35
      }
    }
  }

  app_settings = {
    "SCM_DO_BUILD_DURING_DEPLOYMENT" = "true"
  }

  tags = var.tags
}

# ---------- Azure AD Application / SP / Secret (Single-tenant) ----------
resource "azuread_application" "bot_app" {
  display_name     = var.azuread_app_display_name
  sign_in_audience = "AzureADMyOrg" # single tenant
}

resource "azuread_service_principal" "bot_sp" {
  client_id = azuread_application.bot_app.client_id
}

# 2 anos (17520 horas)
resource "azuread_application_password" "bot_secret" {
  application_id    = azuread_application.bot_app.id
  display_name      = "terraform-generated"
  end_date_relative = "17520h"
}

# ---------- Azure Bot (Single-Tenant) ----------
resource "azurerm_bot_service_azure_bot" "bot" {
  name                = local.bot_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = "global"
  sku                 = var.bot_sku

  display_name = var.bot_display_name

  microsoft_app_id        = azuread_application.bot_app.client_id
  microsoft_app_tenant_id = data.azurerm_client_config.current.tenant_id
  microsoft_app_type      = "SingleTenant"

  endpoint = "https://${azurerm_linux_web_app.webapp.default_hostname}/api/messages"
}

# ---------- Teams Channel ----------
resource "azurerm_bot_channel_ms_teams" "bot_teams" {
  bot_name            = azurerm_bot_service_azure_bot.bot.name
  location            = "global"
  resource_group_name = azurerm_bot_service_azure_bot.bot.resource_group_name
}

# ---------- Exportar credenciais em arquivo local (NÃO versionado) ----------
resource "null_resource" "secrets_dir" {
  provisioner "local-exec" {
    command = "mkdir -p ${path.module}/secrets"
  }
}

resource "local_file" "bot_credentials" {
  filename = "${path.module}/secrets/bot_credentials.json"
  content = jsonencode({
    client_id     = azuread_application.bot_app.client_id
    client_secret = azuread_application_password.bot_secret.value
    tenant_id     = data.azurerm_client_config.current.tenant_id
    endpoint      = "https://${azurerm_linux_web_app.webapp.default_hostname}/api/messages"
  })
  depends_on = [null_resource.secrets_dir]
}
