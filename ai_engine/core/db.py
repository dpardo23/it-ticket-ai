import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Cargar variables del .env
load_dotenv()

def get_supabase_client() -> Client:
    """Retorna una instancia pura del cliente de Supabase."""
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        raise ValueError("Credenciales de Supabase no encontradas en el entorno.")
        
    return create_client(url, key)

# Instancia global (Singleton funcional)
supabase = get_supabase_client()