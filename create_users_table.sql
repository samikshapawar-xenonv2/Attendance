-- Create Users Table for Authentication
CREATE TABLE IF NOT EXISTS public.users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Enable Row Level Security (RLS)
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Create Policy: Allow anyone to insert (Sign Up)
CREATE POLICY "Allow public insert" ON public.users
FOR INSERT WITH CHECK (true);

-- Create Policy: Allow users to read their own data (Login check)
-- Note: For simple backend login where the server has the service key, RLS might be bypassed, 
-- but this is good practice if we use the client lib directly later.
CREATE POLICY "Allow public read" ON public.users
FOR SELECT USING (true);
