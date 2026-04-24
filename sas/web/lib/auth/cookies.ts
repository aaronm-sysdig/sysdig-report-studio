/**
 * Cookie name and helper constants for SAS session auth.
 * The session is cosmetic — not production-grade security.
 */

export const SESSION_COOKIE = "sas_session";

export const COOKIE_OPTIONS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  sameSite: "lax" as const,
  path: "/",
  maxAge: 60 * 60 * 24, // 24 hours in seconds
};
