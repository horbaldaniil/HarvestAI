import { createBrowserRouter, Navigate } from "react-router-dom";
import { FileText } from "lucide-react";

import { LoginPage } from "@/pages/LoginPage";
import { RegisterPage } from "@/pages/RegisterPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { FieldsPage } from "@/pages/FieldsPage";
import { ChatPage } from "@/pages/ChatPage";
import { PlaceholderPage } from "@/pages/PlaceholderPage";
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
        <PlaceholderPage
          title="Звіти"
          description="PDF-звіти по полях з графіками, аномаліями та прогнозом — буде реалізовано на тижні 5."
          icon={<FileText className="h-5 w-5 text-primary" />}
        />
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
