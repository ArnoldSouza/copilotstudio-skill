output "resource_group_name" { value = azurerm_resource_group.rg.name }
output "app_service_plan_name" { value = azurerm_service_plan.asp.name }
output "web_app_name" { value = azurerm_linux_web_app.webapp.name }
output "web_app_default_hostname" { value = azurerm_linux_web_app.webapp.default_hostname }

# Credenciais do bot
output "bot_app_client_id" {
  description = "Microsoft App ID (Client ID) do Bot."
  value       = azuread_application.bot_app.client_id
}

# Caminho do arquivo com o secret (não exibe o secret no console)
output "bot_app_secret_file" {
  description = "Arquivo local (não versionado) contendo o client_secret."
  value       = local_file.bot_credentials.filename
}
