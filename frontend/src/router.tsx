import { createBrowserRouter, Navigate } from "react-router-dom";

import { LoginPage } from "@/pages/LoginPage";
import { RegisterPage } from "@/pages/RegisterPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { FieldsPage } from "@/pages/FieldsPage";
import { ChatPage } from "@/pages/ChatPage";
import { ReportsPage } from "@/pages/ReportsPage";
import { WeatherPage } from "@/pages/WeatherPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { ProtectedRoute } from "@/components/ProtectedRoute";

export const router = createBrowserRouter([
  { path: "/", element: <Navigate to="/dashboard" replace /> },
  { path: "/login", element: <LoginPage /> },
  { path: "/register", element: <RegisterPage /> },
  {
    path: "/dashboard",
    element: (
      <ProtectedRoute>
        <DashboardPage />
      </ProtectedRoute>
    ),
  },
  {
    path: "/fields",
    element: (
      <ProtectedRoute>
        <FieldsPage />
      </ProtectedRoute>
    ),
  },
  {
    path: "/chat",
    element: (
      <ProtectedRoute>
        <ChatPage />
      </ProtectedRoute>
    ),
  },
  {
    path: "/reports",
    element: (
      <ProtectedRoute>
        <ReportsPage />
      </ProtectedRoute>
    ),
  },
  {
    path: "/weather",
    element: (
      <ProtectedRoute>
        <WeatherPage />
      </ProtectedRoute>
    ),
  },
  {
    path: "/settings",
    element: (
      <ProtectedRoute>
        <SettingsPage />
      </ProtectedRoute>
    ),
  },
  { path: "*", element: <Navigate to="/dashboard" replace /> },
]);
