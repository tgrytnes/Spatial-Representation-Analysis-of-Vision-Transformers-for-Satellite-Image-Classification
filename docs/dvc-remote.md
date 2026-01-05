# Azure Blob DVC Remote

Use this remote to keep the EuroSAT data outside of Git while making the dataset available to every collaborator and CI worker.

## Remote configuration
1. Provision an Azure Storage Account (and set up a container) with at least 5 GB of free space and note the `AccountName`.
2. Run:
   ```
   dvc remote add -d azure-remote azure://spatialvit-eurosat/eurosat
   ```
   The path above mirrors the intended container/object layout; adjust it if you choose different names.
3. Provide the Azure connection string without leaking secrets:
   ```
   dvc remote modify azure-remote connection_string "<your-connection-string>"
   ```
   For security, never commit the real connection string. Instead, set it locally:
   ```
   dvc remote modify --local azure-remote connection_string "$AZURE_DVC_CONN"
   ```
   and keep that value in your shell profile, encrypted password manager, or CI secret.

## Data push / pull
- `dvc add data/eurosat` (or replace with your harvest path) tracks the dataset and creates `data/eurosat.dvc`.
- `dvc push --remote azure-remote` uploads the dataset blob to Azure.
- `dvc pull --remote azure-remote` rehydrates the dataset on a clean clone (use `--run-cache` when reproducing the entire pipeline).
- Commit the `.dvc` files and `.dvc/config` (minus any `--local` overrides) so everyone has the same metadata.

## Credential rotation
1. Generate a new SAS token or connection string with the same permissions (read/write) via the Azure Portal or CLI.
2. Update the local configuration:
   ```
   export AZURE_DVC_CONN="DefaultEndpointsProtocol=https;AccountName=…;AccountKey=…;EndpointSuffix=core.windows.net"
   dvc remote modify --local azure-remote connection_string "$AZURE_DVC_CONN"
   ```
3. Broadcast the new secret to the team through a secure channel and expire the old one.

## Sharing access
- Give teammates (or CI) the storage account credentials through encrypted secrets.
- For GitHub Actions use a secret like `AZURE_DVC_CONN` and add:
  ```
  - name: Configure DVC remote
    run: dvc remote modify --local azure-remote connection_string "${{ secrets.AZURE_DVC_CONN }}"
  ```
- Document who owns the connection string and how to contact them in case the storage account needs to be rotated or deleted.
