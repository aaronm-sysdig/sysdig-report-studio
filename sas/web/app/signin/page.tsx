"use client";

import { useState, FormEvent } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function SignInPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const res = await fetch("/api/signin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (res.ok) {
        router.push("/dashboard");
      } else {
        const data = await res.json();
        setError(data.error ?? "Invalid credentials. Please try again.");
      }
    } catch {
      setError("Unable to connect. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="min-h-screen flex items-center justify-center"
      style={{ backgroundColor: "var(--deep-see)" }}
    >
      <div
        className="w-[400px] p-8 shadow-xl"
        style={{ backgroundColor: "var(--white)", borderRadius: "var(--radius)" }}
      >
        {/* Wordmark */}
        <div className="flex items-center gap-2 mb-6">
          <span
            className="text-[22px] font-bold tracking-tight"
            style={{ color: "var(--deep-see)" }}
          >
            sysdig
          </span>
          <span
            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: "var(--lumin)" }}
            aria-hidden="true"
          />
        </div>

        {/* Heading with Lumin left-edge accent */}
        <div className="flex items-center gap-3 mb-6">
          <div
            className="w-1 h-6 rounded-full flex-shrink-0"
            style={{ backgroundColor: "var(--lumin)" }}
            aria-hidden="true"
          />
          <h1
            className="text-xl font-semibold"
            style={{ color: "var(--fg-primary)" }}
          >
            Sign in
          </h1>
        </div>

        <form onSubmit={handleSubmit} noValidate className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="username" style={{ color: "var(--fg-primary)" }}>
              Username
            </Label>
            <Input
              id="username"
              type="text"
              autoComplete="username"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="password" style={{ color: "var(--fg-primary)" }}>
              Password
            </Label>
            <Input
              id="password"
              type="password"
              autoComplete="current-password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {error && (
            <p
              className="text-sm"
              role="alert"
              style={{ color: "var(--severity-critical)" }}
            >
              {error}
            </p>
          )}

          <Button
            type="submit"
            disabled={loading}
            className="w-full font-medium"
            style={{
              backgroundColor: "var(--deep-see)",
              color: "var(--white)",
            }}
          >
            {loading ? "Signing in…" : "Sign in"}
          </Button>
        </form>

        <p
          className="mt-6 text-center text-xs"
          style={{ color: "var(--fg-muted)" }}
        >
          Sysdig Analytics Studio
        </p>
      </div>
    </div>
  );
}
